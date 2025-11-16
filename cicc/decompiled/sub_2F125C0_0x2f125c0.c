// Function: sub_2F125C0
// Address: 0x2f125c0
//
__int64 __fastcall sub_2F125C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  unsigned int *v7; // r13
  unsigned int *v8; // rbx
  unsigned int *v9; // rax
  unsigned int *v10; // rdx
  unsigned int *v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rbx
  unsigned int *v14; // rsi
  unsigned int *v15; // rdi
  int v16; // r14d
  __int64 v17; // rbx
  unsigned int *v18; // rdi
  unsigned int *v19; // rcx
  unsigned int *v20; // rax
  unsigned int *v21; // rdx
  unsigned int *v23; // [rsp+0h] [rbp-80h] BYREF
  __int64 v24; // [rsp+8h] [rbp-78h]
  _BYTE v25[32]; // [rsp+10h] [rbp-70h] BYREF
  unsigned int *v26; // [rsp+30h] [rbp-50h] BYREF
  __int64 v27; // [rsp+38h] [rbp-48h]
  _BYTE v28[64]; // [rsp+40h] [rbp-40h] BYREF

  v6 = 1;
  if ( *(_DWORD *)(a2 + 120) > 1u )
  {
    v7 = *(unsigned int **)(a2 + 152);
    v8 = *(unsigned int **)(a2 + 144);
    if ( v8 != v7 )
    {
      v24 = 0x800000000LL;
      v9 = (unsigned int *)v25;
      v23 = (unsigned int *)v25;
      if ( (unsigned __int64)((char *)v7 - (char *)v8) > 0x20 )
      {
        sub_C8D5F0((__int64)&v23, v25, v7 - v8, 4u, a5, a6);
        v9 = &v23[(unsigned int)v24];
      }
      v10 = v8;
      v11 = (unsigned int *)((char *)v9 + (char *)v7 - (char *)v8);
      do
      {
        if ( v9 )
          *v9 = *v10;
        ++v9;
        ++v10;
      }
      while ( v9 != v11 );
      LODWORD(v24) = v7 - v8 + v24;
      sub_27DE390(v23, &v23[(unsigned int)v24]);
      v13 = (unsigned int)v24;
      v26 = (unsigned int *)v28;
      v14 = (unsigned int *)v28;
      v15 = (unsigned int *)v28;
      v27 = 0x800000000LL;
      v16 = v24;
      if ( (_DWORD)v24 )
      {
        if ( (unsigned int)v24 > 8uLL )
        {
          sub_C8D5F0((__int64)&v26, v28, (unsigned int)v24, 4u, (__int64)&v26, v12);
          v15 = v26;
          v14 = &v26[(unsigned int)v27];
        }
        v17 = v13;
        if ( &v15[v17] != v14 )
        {
          do
          {
            if ( v14 )
              *v14 = -1;
            ++v14;
          }
          while ( &v15[v17] != v14 );
          v15 = v26;
          v14 = &v26[v17];
        }
        LODWORD(v27) = v16;
      }
      sub_27DE390(v15, v14);
      v18 = v23;
      v19 = &v23[(unsigned int)v24];
      if ( v23 == v19 )
      {
LABEL_28:
        v6 = 1;
      }
      else
      {
        v20 = v23;
        v21 = v26;
        while ( *v21 == *v20 )
        {
          ++v20;
          ++v21;
          if ( v20 == v19 )
            goto LABEL_28;
        }
        v6 = 0;
      }
      if ( v26 != (unsigned int *)v28 )
      {
        _libc_free((unsigned __int64)v26);
        v18 = v23;
      }
      if ( v18 != (unsigned int *)v25 )
        _libc_free((unsigned __int64)v18);
    }
  }
  return v6;
}
