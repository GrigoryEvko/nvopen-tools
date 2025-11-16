// Function: sub_CFB0F0
// Address: 0xcfb0f0
//
__int64 __fastcall sub_CFB0F0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  int v3; // edx
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned int v6; // edx
  __int64 v7; // r15
  __int64 v8; // rcx
  _QWORD *v9; // r14
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  bool v14; // zf
  bool v15; // si
  bool v16; // dl
  int v17; // r8d
  __int64 v18; // rcx
  bool v19; // al
  _QWORD v20[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v21; // [rsp+18h] [rbp-78h]
  __int64 v22; // [rsp+20h] [rbp-70h]
  void *v23; // [rsp+30h] [rbp-60h]
  _QWORD v24[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h]

  result = *(_QWORD *)(a1 + 24);
  v20[1] = 0;
  v20[0] = 2;
  v2 = *(_QWORD *)(a1 + 32);
  v21 = result;
  if ( result != -4096 && result != 0 && result != -8192 )
  {
    sub_BD73F0((__int64)v20);
    result = v21;
  }
  v22 = 0;
  v3 = *(_DWORD *)(v2 + 184);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = *(_QWORD *)(v2 + 168);
    v6 = v4 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v7 = v5 + 88LL * v6;
    v8 = *(_QWORD *)(v7 + 24);
    if ( v8 == result )
    {
LABEL_6:
      v9 = *(_QWORD **)(v7 + 40);
      v10 = &v9[4 * *(unsigned int *)(v7 + 48)];
      if ( v9 != v10 )
      {
        do
        {
          v11 = *(v10 - 2);
          v10 -= 4;
          if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
            sub_BD60C0(v10);
        }
        while ( v9 != v10 );
        v10 = *(_QWORD **)(v7 + 40);
      }
      if ( v10 != (_QWORD *)(v7 + 56) )
        _libc_free(v10, v4);
      v25 = -8192;
      v23 = &unk_49DDAE8;
      v26 = 0;
      v12 = *(_QWORD *)(v7 + 24);
      v24[0] = 2;
      v24[1] = 0;
      if ( v12 == -8192 )
      {
        *(_QWORD *)(v7 + 32) = 0;
LABEL_21:
        --*(_DWORD *)(v2 + 176);
        result = v21;
        ++*(_DWORD *)(v2 + 180);
        goto LABEL_22;
      }
      if ( !v12 || v12 == -4096 )
      {
        *(_QWORD *)(v7 + 24) = -8192;
      }
      else
      {
        sub_BD60C0((_QWORD *)(v7 + 8));
        v13 = v25;
        v14 = v25 == -4096;
        *(_QWORD *)(v7 + 24) = v25;
        v15 = v13 != 0;
        v16 = v13 != -8192;
        if ( v14 || v13 == 0 || v13 == -8192 )
        {
          v18 = v26;
          v19 = v15 && v16 && !v14;
LABEL_31:
          *(_QWORD *)(v7 + 32) = v18;
          v23 = &unk_49DB368;
          if ( v19 )
            sub_BD60C0(v24);
          goto LABEL_21;
        }
        sub_BD6050((unsigned __int64 *)(v7 + 8), v24[0] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v18 = v26;
      v19 = v25 != 0 && v25 != -4096 && v25 != -8192;
      goto LABEL_31;
    }
    v17 = 1;
    while ( v8 != -4096 )
    {
      v6 = v4 & (v17 + v6);
      v7 = v5 + 88LL * v6;
      v8 = *(_QWORD *)(v7 + 24);
      if ( v8 == result )
        goto LABEL_6;
      ++v17;
    }
  }
LABEL_22:
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0(v20);
  return result;
}
