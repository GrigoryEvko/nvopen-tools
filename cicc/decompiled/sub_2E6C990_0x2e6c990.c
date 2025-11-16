// Function: sub_2E6C990
// Address: 0x2e6c990
//
__int64 __fastcall sub_2E6C990(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 **v9; // rbx
  __int64 *v10; // rdx
  __int64 v11; // r13
  __int64 **v12; // r15
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 **v15; // rbx
  __int64 **v16; // r13
  __int64 v17; // rsi
  __int64 *v18; // rax
  __int64 *v19; // rdx
  int v20; // ebx
  __int64 *v22; // rax
  __int64 v23; // [rsp-78h] [rbp-78h] BYREF
  __int64 *v24; // [rsp-70h] [rbp-70h]
  __int64 v25; // [rsp-68h] [rbp-68h]
  int v26; // [rsp-60h] [rbp-60h]
  unsigned __int8 v27; // [rsp-5Ch] [rbp-5Ch]
  _BYTE v28[88]; // [rsp-58h] [rbp-58h] BYREF

  v6 = *(_DWORD *)(a1 + 32);
  v7 = 1;
  if ( v6 != *(_DWORD *)(a2 + 32) )
    return 1;
  v8 = *(unsigned int *)(a2 + 16);
  if ( *(_DWORD *)(a1 + 16) != (_DWORD)v8 )
    return (unsigned int)v7;
  v9 = *(__int64 ***)(a2 + 24);
  v10 = (__int64 *)v28;
  v27 = 1;
  v11 = v6;
  v23 = 0;
  v12 = &v9[v11];
  v24 = (__int64 *)v28;
  v25 = 4;
  v26 = 0;
  if ( &v9[v11] == v9 )
  {
    v15 = *(__int64 ***)(a1 + 24);
    v16 = &v15[v11];
    if ( v16 == v15 )
    {
      LODWORD(v7) = 0;
      return (unsigned int)v7;
    }
    while ( 1 )
    {
LABEL_11:
      v17 = **v15;
      if ( (_BYTE)v7 )
      {
        v18 = v24;
        v19 = &v24[HIDWORD(v25)];
        if ( v24 == v19 )
          return (unsigned int)v7;
        while ( v17 != *v18 )
        {
          if ( v19 == ++v18 )
            return (unsigned int)v7;
        }
      }
      else
      {
        v22 = sub_C8CA60((__int64)&v23, v17);
        LODWORD(v7) = v27;
        if ( !v22 )
        {
          v20 = 1;
          goto LABEL_18;
        }
      }
      if ( v16 == ++v15 )
        goto LABEL_17;
    }
  }
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = **v9;
        if ( (_BYTE)v7 )
          break;
LABEL_23:
        ++v9;
        sub_C8CC70((__int64)&v23, v13, (__int64)v10, v8, v7, a6);
        v7 = v27;
        if ( v12 == v9 )
          goto LABEL_10;
      }
      v14 = v24;
      v8 = HIDWORD(v25);
      v10 = &v24[HIDWORD(v25)];
      if ( v24 != v10 )
        break;
LABEL_27:
      if ( HIDWORD(v25) >= (unsigned int)v25 )
        goto LABEL_23;
      v8 = (unsigned int)(HIDWORD(v25) + 1);
      ++v9;
      ++HIDWORD(v25);
      *v10 = v13;
      v7 = v27;
      ++v23;
      if ( v12 == v9 )
        goto LABEL_10;
    }
    while ( v13 != *v14 )
    {
      if ( v10 == ++v14 )
        goto LABEL_27;
    }
    ++v9;
  }
  while ( v12 != v9 );
LABEL_10:
  v15 = *(__int64 ***)(a1 + 24);
  v16 = &v15[*(unsigned int *)(a1 + 32)];
  if ( v16 != v15 )
    goto LABEL_11;
LABEL_17:
  v20 = 0;
LABEL_18:
  if ( !(_BYTE)v7 )
    _libc_free((unsigned __int64)v24);
  LODWORD(v7) = v20;
  return (unsigned int)v7;
}
