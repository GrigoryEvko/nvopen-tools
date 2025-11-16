// Function: sub_A78500
// Address: 0xa78500
//
unsigned __int64 __fastcall sub_A78500(__int64 *a1, unsigned __int64 *a2, int a3, unsigned __int64 a4)
{
  unsigned int v4; // r14d
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 *v8; // rdi
  unsigned __int64 *v9; // r10
  int v10; // edx
  signed __int64 v11; // r8
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // r9
  unsigned __int64 *v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 *v17; // rdi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r12
  unsigned __int64 v21; // r14
  unsigned __int64 *v22; // rax
  unsigned __int64 *v23; // rcx
  signed __int64 v24; // [rsp+8h] [rbp-88h]
  unsigned __int64 *v25; // [rsp+10h] [rbp-80h]
  __int64 v26; // [rsp+10h] [rbp-80h]
  unsigned __int64 *v27; // [rsp+18h] [rbp-78h]
  unsigned __int64 *v29; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30; // [rsp+38h] [rbp-58h]
  _BYTE dest[80]; // [rsp+40h] [rbp-50h] BYREF

  v4 = a3 + 1;
  v27 = a2;
  v6 = sub_A74460(a1);
  v7 = sub_A74450(a1);
  v29 = (unsigned __int64 *)dest;
  v8 = (unsigned __int64 *)dest;
  v9 = (unsigned __int64 *)v7;
  v10 = 0;
  v11 = v6 - v7;
  v30 = 0x400000000LL;
  v12 = (v6 - v7) >> 3;
  if ( (unsigned __int64)(v6 - v7) > 0x20 )
  {
    a2 = (unsigned __int64 *)dest;
    v24 = v6 - v7;
    v25 = (unsigned __int64 *)v7;
    sub_C8D5F0(&v29, dest, v11 >> 3, 8);
    v10 = v30;
    v11 = v24;
    v9 = v25;
    v8 = &v29[(unsigned int)v30];
  }
  if ( (unsigned __int64 *)v6 != v9 )
  {
    a2 = v9;
    memcpy(v8, v9, v11);
    v10 = v30;
  }
  LODWORD(v13) = v12 + v10;
  v14 = v4;
  LODWORD(v30) = v12 + v10;
  if ( v4 < (int)v12 + v10 )
    goto LABEL_6;
  v21 = (unsigned int)(a3 + 2);
  v13 = (unsigned int)v13;
  if ( v21 == (unsigned int)v13 )
    goto LABEL_6;
  if ( v21 < (unsigned int)v13 )
  {
    LODWORD(v30) = a3 + 2;
LABEL_6:
    v15 = v29;
    goto LABEL_7;
  }
  if ( v21 > HIDWORD(v30) )
  {
    a2 = (unsigned __int64 *)dest;
    v26 = v14;
    sub_C8D5F0(&v29, dest, v21, 8);
    v13 = (unsigned int)v30;
    v14 = v26;
  }
  v15 = v29;
  v22 = &v29[v13];
  v23 = &v29[v21];
  if ( v22 != v23 )
  {
    do
    {
      if ( v22 )
        *v22 = 0;
      ++v22;
    }
    while ( v23 != v22 );
    v15 = v29;
  }
  LODWORD(v30) = a3 + 2;
LABEL_7:
  v15[v14] = a4;
  v16 = (unsigned int)v30;
  v17 = v29;
  if ( (_DWORD)v30 )
  {
    while ( !v29[v16 - 1] )
    {
      LODWORD(v30) = --v16;
      if ( !v16 )
        goto LABEL_12;
    }
    a2 = v29;
    v18 = sub_A77EC0(v27, v29, v16);
    v17 = v29;
    v19 = v18;
  }
  else
  {
LABEL_12:
    v19 = 0;
  }
  if ( v17 != (unsigned __int64 *)dest )
    _libc_free(v17, a2);
  return v19;
}
