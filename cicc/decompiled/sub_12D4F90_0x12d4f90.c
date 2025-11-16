// Function: sub_12D4F90
// Address: 0x12d4f90
//
__int64 __fastcall sub_12D4F90(_QWORD *a1)
{
  __int64 v1; // rsi
  __int64 result; // rax
  __int64 v3; // rbx
  int v4; // edx
  int v5; // ecx
  __int64 v6; // rdi
  unsigned int v7; // edx
  _QWORD *v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rcx
  bool v15; // al
  int v16; // r8d
  _QWORD v17[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+20h] [rbp-60h]
  void *v20; // [rsp+30h] [rbp-50h]
  _QWORD v21[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v22; // [rsp+48h] [rbp-38h]
  __int64 v23; // [rsp+50h] [rbp-30h]

  v1 = a1[1];
  v17[1] = 0;
  v17[0] = v1 & 6;
  v18 = a1[3];
  result = v18;
  if ( v18 != 0 && v18 != -8 && v18 != -16 )
  {
    sub_1649AC0(v17, v1 & 0xFFFFFFFFFFFFFFF8LL);
    result = v18;
  }
  v3 = a1[4];
  v19 = v3;
  v4 = *(_DWORD *)(v3 + 24);
  if ( !v4 )
    goto LABEL_5;
  v5 = v4 - 1;
  v6 = *(_QWORD *)(v3 + 8);
  v7 = (v4 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v8 = (_QWORD *)(v6 + ((unsigned __int64)v7 << 6));
  v9 = v8[3];
  if ( v9 == result )
  {
LABEL_10:
    v10 = v8[7];
    if ( v10 != -8 && v10 != 0 && v10 != -16 )
      sub_1649B30(v8 + 5);
    v22 = -16;
    v20 = &unk_49E6B50;
    v23 = 0;
    v11 = v8[3];
    v21[0] = 2;
    v21[1] = 0;
    if ( v11 == -16 )
    {
      v8[4] = 0;
    }
    else
    {
      if ( !v11 || v11 == -8 )
      {
        v8[3] = -16;
        v14 = v23;
        v15 = v22 != -16 && v22 != 0 && v22 != -8;
LABEL_19:
        v8[4] = v14;
        v20 = &unk_49EE2B0;
        if ( v15 )
          sub_1649B30(v21);
        goto LABEL_21;
      }
      sub_1649B30(v8 + 1);
      v12 = v22;
      v13 = v22 == -8;
      v8[3] = v22;
      if ( v12 != 0 && !v13 && v12 != -16 )
      {
        sub_1649AC0(v8 + 1, v21[0] & 0xFFFFFFFFFFFFFFF8LL);
        v14 = v23;
        v15 = v22 != 0 && v22 != -16 && v22 != -8;
        goto LABEL_19;
      }
      v8[4] = v23;
    }
LABEL_21:
    --*(_DWORD *)(v3 + 16);
    ++*(_DWORD *)(v3 + 20);
    result = v18;
    goto LABEL_5;
  }
  v16 = 1;
  while ( v9 != -8 )
  {
    v7 = v5 & (v16 + v7);
    v8 = (_QWORD *)(v6 + ((unsigned __int64)v7 << 6));
    v9 = v8[3];
    if ( v9 == result )
      goto LABEL_10;
    ++v16;
  }
LABEL_5:
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(v17);
  return result;
}
