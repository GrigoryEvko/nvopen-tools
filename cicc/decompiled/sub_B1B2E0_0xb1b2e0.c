// Function: sub_B1B2E0
// Address: 0xb1b2e0
//
__int64 __fastcall sub_B1B2E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 *v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 *v9; // r14
  __int64 v10; // rcx
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // [rsp+0h] [rbp-60h] BYREF
  _BYTE *v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  int v19; // [rsp+18h] [rbp-48h]
  unsigned __int8 v20; // [rsp+1Ch] [rbp-44h]
  _BYTE v21[64]; // [rsp+20h] [rbp-40h] BYREF

  v2 = 0;
  v3 = *(unsigned int *)(a1 + 8);
  if ( v3 != *(_DWORD *)(a2 + 8) )
    return v2;
  v5 = *(__int64 **)a1;
  v6 = v3;
  v7 = v21;
  v20 = 1;
  v16 = 0;
  v9 = &v5[v6];
  v17 = v21;
  v18 = 4;
  v19 = 0;
  if ( v5 != &v5[v6] )
  {
    v10 = 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *v5;
        if ( (_BYTE)v10 )
          break;
LABEL_21:
        ++v5;
        sub_C8CC70(&v16, v11);
        v10 = v20;
        if ( v9 == v5 )
          goto LABEL_11;
      }
      v12 = v17;
      v7 = &v17[8 * HIDWORD(v18)];
      if ( v17 == (_BYTE *)v7 )
      {
LABEL_27:
        if ( HIDWORD(v18) >= (unsigned int)v18 )
          goto LABEL_21;
        ++v5;
        ++HIDWORD(v18);
        *v7 = v11;
        v10 = v20;
        ++v16;
        if ( v9 == v5 )
          goto LABEL_11;
      }
      else
      {
        while ( v11 != *v12 )
        {
          if ( v7 == ++v12 )
            goto LABEL_27;
        }
        if ( v9 == ++v5 )
        {
LABEL_11:
          v13 = *(__int64 **)a2;
          v14 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
          if ( v14 != *(_QWORD *)a2 )
            goto LABEL_12;
LABEL_18:
          v2 = 1;
LABEL_19:
          if ( !(_BYTE)v10 )
          {
            _libc_free(v17, v11);
            return v2;
          }
          return v2;
        }
      }
    }
  }
  v13 = *(__int64 **)a2;
  v14 = *(_QWORD *)a2 + v6 * 8;
  if ( v14 == *(_QWORD *)a2 )
    return 1;
  v10 = 1;
  while ( 1 )
  {
LABEL_12:
    while ( 1 )
    {
      v11 = *v13;
      if ( (_BYTE)v10 )
        break;
      if ( !sub_C8CA60(&v16, v11, v7, v10) )
      {
        LOBYTE(v10) = v20;
        v2 = 0;
        goto LABEL_19;
      }
      ++v13;
      v10 = v20;
      if ( (__int64 *)v14 == v13 )
        goto LABEL_18;
    }
    v15 = v17;
    v7 = &v17[8 * HIDWORD(v18)];
    if ( v17 == (_BYTE *)v7 )
      return 0;
    while ( v11 != *v15 )
    {
      if ( v7 == ++v15 )
        return 0;
    }
    if ( (__int64 *)v14 == ++v13 )
      goto LABEL_18;
  }
}
