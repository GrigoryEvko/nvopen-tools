// Function: sub_2464C00
// Address: 0x2464c00
//
__int64 __fastcall sub_2464C00(__int64 *a1, unsigned int **a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  __int64 v8; // rdi
  unsigned int v9; // r15d
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // r15
  __int64 v13; // rcx
  __int64 *v14; // rax
  unsigned __int64 v15; // rdx
  __int64 *v16; // r14
  __int64 *v17; // rax
  char v18; // si
  __int64 v19; // rsi
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // r15
  _BYTE *v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r15
  const char *v27; // rax
  __int64 *v28; // rbx
  __int64 *v29; // rdx
  __int64 *v30; // r14
  __int64 *v31; // rax
  char v32; // si
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v37; // [rsp+10h] [rbp-C0h]
  unsigned int v38; // [rsp+1Ch] [rbp-B4h]
  unsigned __int64 v40; // [rsp+38h] [rbp-98h]
  _BYTE v41[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v42; // [rsp+60h] [rbp-70h]
  const char *v43; // [rsp+70h] [rbp-60h] BYREF
  __int64 v44; // [rsp+78h] [rbp-58h]
  _QWORD v45[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v46; // [rsp+90h] [rbp-40h]

  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(_DWORD *)(v8 + 32);
  v42 = 257;
  v38 = v9;
  v37 = sub_AD6530(v8, (__int64)a2);
  v43 = (const char *)v45;
  v40 = v9;
  v44 = 0x400000000LL;
  if ( !v9 )
  {
    v12 = v45;
    v19 = 0;
    goto LABEL_12;
  }
  if ( v9 <= 4uLL )
  {
    v12 = v45;
    v13 = 8 * v40;
    v14 = &v45[v40];
    do
    {
LABEL_4:
      if ( v12 )
        *v12 = 0;
      ++v12;
    }
    while ( v14 != v12 );
    v15 = (unsigned __int64)v43;
    v12 = (__int64 *)&v43[v13];
    goto LABEL_8;
  }
  sub_C8D5F0((__int64)&v43, v45, v9, 8u, v10, v11);
  v15 = (unsigned __int64)v43;
  v13 = 8LL * v9;
  v12 = (__int64 *)&v43[8 * (unsigned int)v44];
  v14 = (__int64 *)&v43[v13];
  if ( &v43[v13] != (const char *)v12 )
    goto LABEL_4;
LABEL_8:
  LODWORD(v44) = v38;
  if ( (__int64 *)v15 == v12 )
  {
    v19 = v40;
  }
  else
  {
    v16 = (__int64 *)v15;
    do
    {
      ++v16;
      v17 = (__int64 *)sub_B2BE50(*a1);
      v18 = a4;
      a4 >>= 1;
      *(v16 - 1) = sub_ACD760(v17, v18 & 1);
    }
    while ( v12 != v16 );
    v12 = (__int64 *)v43;
    v19 = (unsigned int)v44;
  }
LABEL_12:
  v20 = sub_AD3730(v12, v19);
  if ( v43 != (const char *)v45 )
    _libc_free((unsigned __int64)v43);
  v21 = sub_B36550(a2, v20, a3, v37, (__int64)v41, 0);
  v22 = sub_B34870((__int64)a2, v21);
  v46 = 259;
  v43 = "_msdpp";
  v23 = (_BYTE *)sub_AD6530(*(_QWORD *)(v22 + 8), v21);
  v26 = sub_92B530(a2, 0x20u, v22, v23, (__int64)&v43);
  v27 = (const char *)v45;
  v44 = 0x400000000LL;
  v43 = (const char *)v45;
  v28 = v45;
  if ( v40 )
  {
    if ( v40 > 4 )
    {
      sub_C8D5F0((__int64)&v43, v45, v40, 8u, v24, v25);
      v27 = v43;
      v28 = (__int64 *)&v43[8 * (unsigned int)v44];
    }
    v29 = (__int64 *)&v27[8 * v40];
    if ( v29 != v28 )
    {
      do
      {
        if ( v28 )
          *v28 = 0;
        ++v28;
      }
      while ( v29 != v28 );
      v27 = v43;
      v28 = (__int64 *)&v43[8 * v40];
    }
    LODWORD(v44) = v38;
    if ( v27 != (const char *)v28 )
    {
      v30 = (__int64 *)v27;
      do
      {
        ++v30;
        v31 = (__int64 *)sub_B2BE50(*a1);
        v32 = a5;
        a5 >>= 1;
        *(v30 - 1) = sub_ACD760(v31, v32 & 1);
      }
      while ( v28 != v30 );
      v28 = (__int64 *)v43;
      v40 = (unsigned int)v44;
    }
  }
  v33 = sub_AD3730(v28, v40);
  if ( v43 != (const char *)v45 )
    _libc_free((unsigned __int64)v43);
  v46 = 257;
  v34 = sub_AD6530(*(_QWORD *)(v33 + 8), v40);
  return sub_B36550(a2, v26, v34, v33, (__int64)&v43, 0);
}
