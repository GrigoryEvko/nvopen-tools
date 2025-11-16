// Function: sub_1E10A60
// Address: 0x1e10a60
//
__int64 __fastcall sub_1E10A60(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rdi
  unsigned int v8; // ecx
  _QWORD *v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  bool v16; // al
  bool v17; // zf
  int v18; // r10d
  __int64 v19; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+8h] [rbp-98h]
  void *v21; // [rsp+10h] [rbp-90h]
  unsigned __int64 v22[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v23; // [rsp+28h] [rbp-78h]
  __int64 v24; // [rsp+30h] [rbp-70h]
  void *v25; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v26[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v27; // [rsp+58h] [rbp-48h]
  __int64 v28; // [rsp+60h] [rbp-40h]

  v3 = a1[1];
  v22[1] = 0;
  v22[0] = v3 & 6;
  v23 = a1[3];
  result = v23;
  if ( v23 != 0 && v23 != -8 && v23 != -16 )
  {
    sub_1649AC0(v22, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v23;
  }
  v5 = a1[4];
  v24 = v5;
  v21 = &unk_49FB768;
  v6 = *(unsigned int *)(v5 + 24);
  if ( !(_DWORD)v6 )
    goto LABEL_5;
  v7 = *(_QWORD *)(v5 + 8);
  v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v9 = (_QWORD *)(v7 + 48LL * v8);
  v10 = v9[3];
  if ( v10 == result )
  {
LABEL_10:
    if ( v9 == (_QWORD *)(v7 + 48 * v6) )
      goto LABEL_5;
    v11 = v9[5];
    v26[0] = 2;
    v26[1] = 0;
    v27 = -16;
    v25 = &unk_49FB768;
    v28 = 0;
    v12 = v9[3];
    if ( v12 == -16 )
    {
      v9[4] = 0;
    }
    else
    {
      if ( !v12 || v12 == -8 )
      {
        v9[3] = -16;
        v14 = v27;
        v15 = v28;
        v16 = v27 != -8;
        v17 = v27 == 0;
LABEL_17:
        v9[4] = v15;
        v25 = &unk_49EE2B0;
        if ( v14 != -16 && !v17 && v16 )
          sub_1649B30(v26);
        goto LABEL_19;
      }
      sub_1649B30(v9 + 1);
      v13 = v27;
      v17 = v27 == 0;
      v9[3] = v27;
      if ( v13 != -8 && !v17 && v13 != -16 )
      {
        sub_1649AC0(v9 + 1, v26[0] & 0xFFFFFFFFFFFFFFF8LL);
        v14 = v27;
        v15 = v28;
        v16 = v27 != 0;
        v17 = v27 == -8;
        goto LABEL_17;
      }
      v9[4] = v28;
    }
LABEL_19:
    --*(_DWORD *)(v5 + 16);
    ++*(_DWORD *)(v5 + 20);
    v19 = a2;
    v20 = v11;
    sub_1E105B0((__int64)&v25, v24, &v19);
    if ( v20 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 16LL))(v20);
    result = v23;
    goto LABEL_5;
  }
  v18 = 1;
  while ( v10 != -8 )
  {
    v8 = (v6 - 1) & (v18 + v8);
    v9 = (_QWORD *)(v7 + 48LL * v8);
    v10 = v9[3];
    if ( v10 == result )
      goto LABEL_10;
    ++v18;
  }
LABEL_5:
  v21 = &unk_49EE2B0;
  if ( result != 0 && result != -8 && result != -16 )
    return sub_1649B30(v22);
  return result;
}
