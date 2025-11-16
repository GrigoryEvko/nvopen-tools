// Function: sub_2CF9900
// Address: 0x2cf9900
//
__int64 __fastcall sub_2CF9900(_QWORD *a1)
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
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rcx
  bool v14; // al
  int v15; // r9d
  unsigned __int64 v16[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v17; // [rsp+18h] [rbp-68h]
  __int64 v18; // [rsp+20h] [rbp-60h]
  void *v19; // [rsp+30h] [rbp-50h]
  _QWORD v20[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v21; // [rsp+48h] [rbp-38h]
  __int64 v22; // [rsp+50h] [rbp-30h]

  v1 = a1[1];
  v16[1] = 0;
  v16[0] = v1 & 6;
  v17 = a1[3];
  result = v17;
  if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
  {
    sub_BD6050(v16, v1 & 0xFFFFFFFFFFFFFFF8LL);
    result = v17;
  }
  v3 = a1[4];
  v18 = v3;
  v4 = *(_DWORD *)(v3 + 24);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = *(_QWORD *)(v3 + 8);
    v7 = (v4 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v8 = (_QWORD *)(v6 + 48LL * v7);
    v9 = v8[3];
    if ( v9 == result )
    {
LABEL_6:
      v21 = -8192;
      v19 = &unk_4A259B8;
      v22 = 0;
      v10 = v8[3];
      v20[0] = 2;
      v20[1] = 0;
      if ( v10 == -8192 )
      {
        v8[4] = 0;
      }
      else
      {
        if ( !v10 || v10 == -4096 )
        {
          v8[3] = -8192;
          v13 = v22;
          v14 = v21 != -8192 && v21 != 0 && v21 != -4096;
LABEL_22:
          v8[4] = v13;
          v19 = &unk_49DB368;
          if ( v14 )
            sub_BD60C0(v20);
          goto LABEL_13;
        }
        sub_BD60C0(v8 + 1);
        v11 = v21;
        v12 = v21 == -4096;
        v8[3] = v21;
        if ( v11 != 0 && !v12 && v11 != -8192 )
        {
          sub_BD6050(v8 + 1, v20[0] & 0xFFFFFFFFFFFFFFF8LL);
          v13 = v22;
          v14 = v21 != 0 && v21 != -8192 && v21 != -4096;
          goto LABEL_22;
        }
        v8[4] = v22;
      }
LABEL_13:
      --*(_DWORD *)(v3 + 16);
      ++*(_DWORD *)(v3 + 20);
      result = v17;
      goto LABEL_14;
    }
    v15 = 1;
    while ( v9 != -4096 )
    {
      v7 = v5 & (v15 + v7);
      v8 = (_QWORD *)(v6 + 48LL * v7);
      v9 = v8[3];
      if ( v9 == result )
        goto LABEL_6;
      ++v15;
    }
  }
LABEL_14:
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0(v16);
  return result;
}
