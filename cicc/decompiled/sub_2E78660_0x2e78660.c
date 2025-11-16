// Function: sub_2E78660
// Address: 0x2e78660
//
__int64 __fastcall sub_2E78660(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // r13
  int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  bool v18; // zf
  __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-78h] BYREF
  __int64 v21; // [rsp+10h] [rbp-70h]
  __int64 v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  void *v24; // [rsp+30h] [rbp-50h]
  _QWORD v25[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v26; // [rsp+48h] [rbp-38h]
  __int64 v27; // [rsp+50h] [rbp-30h]

  v6 = a1[1];
  v21 = 0;
  v20 = v6 & 6;
  v22 = a1[3];
  result = v22;
  if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
  {
    sub_BD6050((unsigned __int64 *)&v20, v6 & 0xFFFFFFFFFFFFFFF8LL);
    result = v22;
  }
  v8 = a1[4];
  v23 = v8;
  v9 = *(_DWORD *)(v8 + 24);
  if ( v9 )
  {
    v10 = (unsigned int)(v9 - 1);
    v11 = *(_QWORD *)(v8 + 8);
    v12 = (unsigned int)v10 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v13 = (_QWORD *)(v11 + 48 * v12);
    v14 = v13[3];
    if ( v14 == result )
    {
LABEL_6:
      v15 = v13[5];
      if ( v15 )
        (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, void *, __int64, __int64))(*(_QWORD *)v15 + 16LL))(
          v15,
          v14,
          v12,
          v10,
          a5,
          a6,
          &unk_4A28E90,
          v20,
          v21);
      v26 = -8192;
      v24 = &unk_4A28E90;
      v27 = 0;
      v16 = v13[3];
      v25[0] = 2;
      v25[1] = 0;
      if ( v16 == -8192 )
      {
        v13[4] = 0;
      }
      else
      {
        if ( !v16 || v16 == -4096 )
        {
          v13[3] = -8192;
LABEL_24:
          v19 = v26;
          v18 = v26 == 0;
          v13[4] = v27;
          v24 = &unk_49DB368;
          if ( v19 != -8192 && v19 != -4096 && !v18 )
            sub_BD60C0(v25);
          goto LABEL_15;
        }
        sub_BD60C0(v13 + 1);
        v17 = v26;
        v18 = v26 == 0;
        v13[3] = v26;
        if ( v17 != -4096 && !v18 && v17 != -8192 )
        {
          sub_BD6050(v13 + 1, v25[0] & 0xFFFFFFFFFFFFFFF8LL);
          goto LABEL_24;
        }
        v13[4] = v27;
      }
LABEL_15:
      --*(_DWORD *)(v8 + 16);
      ++*(_DWORD *)(v8 + 20);
      result = v22;
      goto LABEL_16;
    }
    a5 = 1;
    while ( v14 != -4096 )
    {
      a6 = (unsigned int)(a5 + 1);
      v12 = (unsigned int)v10 & ((_DWORD)a5 + (_DWORD)v12);
      v13 = (_QWORD *)(v11 + 48LL * (unsigned int)v12);
      v14 = v13[3];
      if ( v14 == result )
        goto LABEL_6;
      a5 = (unsigned int)a6;
    }
  }
LABEL_16:
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0(&v20);
  return result;
}
