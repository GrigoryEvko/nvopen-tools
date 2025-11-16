// Function: sub_215D780
// Address: 0x215d780
//
__int64 __fastcall sub_215D780(_QWORD *a1)
{
  __int64 v1; // rsi
  __int64 result; // rax
  __int64 v3; // rbx
  int v4; // edx
  int v5; // ecx
  __int64 v6; // r8
  unsigned int v7; // edx
  _QWORD *v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rcx
  bool v12; // zf
  bool v13; // al
  bool v14; // dl
  __int64 v15; // rcx
  bool v16; // al
  int v17; // r9d
  unsigned __int64 v18[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+20h] [rbp-60h]
  void *v21; // [rsp+30h] [rbp-50h]
  _QWORD v22[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v23; // [rsp+48h] [rbp-38h]
  __int64 v24; // [rsp+50h] [rbp-30h]

  v1 = a1[1];
  v18[1] = 0;
  v18[0] = v1 & 6;
  v19 = a1[3];
  result = v19;
  if ( v19 != 0 && v19 != -8 && v19 != -16 )
  {
    sub_1649AC0(v18, v1 & 0xFFFFFFFFFFFFFFF8LL);
    result = v19;
  }
  v3 = a1[4];
  v20 = v3;
  v4 = *(_DWORD *)(v3 + 24);
  if ( !v4 )
    goto LABEL_5;
  v5 = v4 - 1;
  v6 = *(_QWORD *)(v3 + 8);
  v7 = (v4 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v8 = (_QWORD *)(v6 + 48LL * v7);
  v9 = v8[3];
  if ( result == v9 )
  {
LABEL_10:
    v23 = -16;
    v21 = &unk_4A01B30;
    v24 = 0;
    v10 = v8[3];
    v22[0] = 2;
    v22[1] = 0;
    if ( v10 == -16 )
    {
      v8[4] = 0;
LABEL_19:
      --*(_DWORD *)(v3 + 16);
      ++*(_DWORD *)(v3 + 20);
      result = v19;
      goto LABEL_5;
    }
    if ( !v10 || v10 == -8 )
    {
      v8[3] = -16;
    }
    else
    {
      sub_1649B30(v8 + 1);
      v11 = v23;
      v12 = v23 == -8;
      v8[3] = v23;
      v13 = v11 != 0;
      v14 = v11 != -16;
      if ( v11 == 0 || v12 || v11 == -16 )
      {
        v15 = v24;
        v16 = !v12 && v14 && v13;
LABEL_17:
        v8[4] = v15;
        v21 = &unk_49EE2B0;
        if ( v16 )
          sub_1649B30(v22);
        goto LABEL_19;
      }
      sub_1649AC0(v8 + 1, v22[0] & 0xFFFFFFFFFFFFFFF8LL);
    }
    v15 = v24;
    v16 = v23 != -8 && v23 != -16 && v23 != 0;
    goto LABEL_17;
  }
  v17 = 1;
  while ( v9 != -8 )
  {
    v7 = v5 & (v17 + v7);
    v8 = (_QWORD *)(v6 + 48LL * v7);
    v9 = v8[3];
    if ( v9 == result )
      goto LABEL_10;
    ++v17;
  }
LABEL_5:
  if ( result != 0 && result != -8 && result != -16 )
    return sub_1649B30(v18);
  return result;
}
