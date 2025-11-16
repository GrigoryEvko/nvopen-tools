// Function: sub_1892880
// Address: 0x1892880
//
__int64 __fastcall sub_1892880(_QWORD *a1, __int64 a2)
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
  bool v14; // zf
  __int64 v15; // rdx
  int v16; // r10d
  __int64 v17[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 (__fastcall **v18)(); // [rsp+10h] [rbp-90h]
  unsigned __int64 v19[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v20; // [rsp+28h] [rbp-78h]
  __int64 v21; // [rsp+30h] [rbp-70h]
  __int64 (__fastcall **v22)(); // [rsp+40h] [rbp-60h] BYREF
  _QWORD v23[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v24; // [rsp+58h] [rbp-48h]
  __int64 v25; // [rsp+60h] [rbp-40h]

  v3 = a1[1];
  v19[1] = 0;
  v19[0] = v3 & 6;
  v20 = a1[3];
  result = v20;
  if ( v20 != 0 && v20 != -8 && v20 != -16 )
  {
    sub_1649AC0(v19, v3 & 0xFFFFFFFFFFFFFFF8LL);
    result = v20;
  }
  v5 = a1[4];
  v18 = off_49F1D90;
  v21 = v5;
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
    v23[0] = 2;
    v23[1] = 0;
    v24 = -16;
    v22 = off_49F1D90;
    v25 = 0;
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
LABEL_17:
        v15 = v24;
        v14 = v24 == 0;
        v9[4] = v25;
        v22 = (__int64 (__fastcall **)())&unk_49EE2B0;
        if ( v15 != -16 && v15 != -8 && !v14 )
          sub_1649B30(v23);
        goto LABEL_19;
      }
      sub_1649B30(v9 + 1);
      v13 = v24;
      v14 = v24 == 0;
      v9[3] = v24;
      if ( v13 != -8 && !v14 && v13 != -16 )
      {
        sub_1649AC0(v9 + 1, v23[0] & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_17;
      }
      v9[4] = v25;
    }
LABEL_19:
    --*(_DWORD *)(v5 + 16);
    ++*(_DWORD *)(v5 + 20);
    v17[0] = a2;
    v17[1] = v11;
    sub_18923E0((__int64)&v22, v21, v17);
    result = v20;
    goto LABEL_5;
  }
  v16 = 1;
  while ( v10 != -8 )
  {
    v8 = (v6 - 1) & (v16 + v8);
    v9 = (_QWORD *)(v7 + 48LL * v8);
    v10 = v9[3];
    if ( v10 == result )
      goto LABEL_10;
    ++v16;
  }
LABEL_5:
  v18 = (__int64 (__fastcall **)())&unk_49EE2B0;
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(v19);
  return result;
}
