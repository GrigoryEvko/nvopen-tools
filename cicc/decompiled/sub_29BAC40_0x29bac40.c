// Function: sub_29BAC40
// Address: 0x29bac40
//
double __fastcall sub_29BAC40(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v6; // r10
  __int64 *v9; // r12
  unsigned int v11; // edx
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // r15d
  unsigned int v23; // r15d
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 *v26; // r11
  char *v27; // r10
  _QWORD *v28; // r9
  __int64 *v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // rbx
  _QWORD *v32; // r11
  int v34; // r8d
  __int64 v35; // r9
  __int64 v36; // [rsp+0h] [rbp-C0h]
  __int64 v37; // [rsp+0h] [rbp-C0h]
  double v38; // [rsp+8h] [rbp-B8h]
  int v39; // [rsp+8h] [rbp-B8h]
  void *s; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v41; // [rsp+18h] [rbp-A8h]
  _BYTE v42[48]; // [rsp+20h] [rbp-A0h] BYREF
  void *v43; // [rsp+50h] [rbp-70h] BYREF
  __int64 v44; // [rsp+58h] [rbp-68h]
  _BYTE v45[96]; // [rsp+60h] [rbp-60h] BYREF

  v6 = a2;
  v9 = a1;
  s = v42;
  v41 = 0x600000000LL;
  if ( a4 > 6 )
  {
    v36 = a6;
    sub_C8D5F0((__int64)&s, v42, a4, 8u, a5, a6);
    v6 = a2;
    a6 = v36;
    if ( 8 * a4 )
    {
      memset(s, 0, 8 * a4);
      v6 = a2;
      a6 = v36;
    }
    LODWORD(v41) = a4;
    v17 = (unsigned int)a4;
    if ( v6 <= 1 )
    {
      v43 = v45;
      v44 = 0x600000000LL;
      goto LABEL_31;
    }
  }
  else
  {
    if ( a4 )
    {
      v11 = 8 * a4;
      if ( 8 * a4 )
      {
        v12 = v11;
        v13 = v11 - 1;
        *(_QWORD *)&v42[v12 - 8] = 0;
        if ( v13 >= 8 )
        {
          v14 = v13 & 0xFFFFFFF8;
          v15 = 0;
          do
          {
            v16 = v15;
            v15 += 8;
            *(_QWORD *)&v42[v16] = 0;
          }
          while ( v15 < v14 );
        }
      }
    }
    LODWORD(v41) = a4;
    v17 = (unsigned int)a4;
    if ( a2 <= 1 )
    {
      HIDWORD(v44) = 6;
      v43 = v45;
LABEL_11:
      if ( a4 )
      {
        v20 = 8 * a4;
        if ( v20 )
        {
          v21 = (unsigned int)v20;
          v22 = v20 - 1;
          *(_QWORD *)&v45[v21 - 8] = 0;
          if ( v22 >= 8 )
          {
            v23 = v22 & 0xFFFFFFF8;
            v24 = 0;
            do
            {
              v25 = v24;
              v24 += 8;
              *(_QWORD *)&v45[v25] = 0;
            }
            while ( v24 < v23 );
          }
        }
      }
      LODWORD(v44) = v17;
      v26 = (__int64 *)(a5 + 24 * a6);
      if ( (__int64 *)a5 == v26 )
      {
        v38 = 0.0;
        goto LABEL_24;
      }
      v27 = v45;
      goto LABEL_18;
    }
  }
  v18 = (__int64)&a1[v6 - 1];
  do
  {
    v19 = *v9++;
    *((_QWORD *)s + *v9) = *((_QWORD *)s + v19) + *(_QWORD *)(a3 + 8 * v19);
  }
  while ( (__int64 *)v18 != v9 );
  v43 = v45;
  v44 = 0x600000000LL;
  if ( a4 <= 6 )
    goto LABEL_11;
LABEL_31:
  v37 = a6;
  v39 = v17;
  sub_C8D5F0((__int64)&v43, v45, a4, 8u, v17, a6);
  v34 = v39;
  v35 = v37;
  v27 = (char *)v43 + 8 * a4;
  if ( v43 != v27 )
  {
    memset(v43, 0, 8 * a4);
    v27 = (char *)v43;
    v35 = v37;
    v34 = v39;
  }
  LODWORD(v44) = v34;
  v26 = (__int64 *)(a5 + 24 * v35);
  if ( v26 == (__int64 *)a5 )
  {
    v38 = 0.0;
    goto LABEL_22;
  }
LABEL_18:
  v28 = (_QWORD *)a5;
  v29 = (__int64 *)a5;
  do
  {
    v30 = *v29;
    v29 += 3;
    ++*(_QWORD *)&v27[8 * v30];
    v27 = (char *)v43;
  }
  while ( v26 != v29 );
  v31 = s;
  v38 = 0.0;
  do
    v38 = sub_29B8240(v31[*v28], *(_QWORD *)(a3 + 8LL * *v28), v31[v28[1]], v28[2], *(_QWORD *)&v27[8 * *v28] > 1u)
        + v38;
  while ( v32 != v28 );
LABEL_22:
  if ( v27 != v45 )
    _libc_free((unsigned __int64)v27);
LABEL_24:
  if ( s != v42 )
    _libc_free((unsigned __int64)s);
  return v38;
}
