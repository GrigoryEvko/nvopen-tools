// Function: sub_10024E0
// Address: 0x10024e0
//
unsigned __int8 *__fastcall sub_10024E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // r12
  __int64 v7; // r15
  unsigned __int64 v8; // rcx
  _QWORD *v9; // rax
  __int64 *v10; // rdi
  _BYTE *v11; // rdx
  int v12; // ebx
  __int64 v13; // rsi
  _QWORD *i; // rdx
  __int64 v15; // rbx
  unsigned __int8 *v16; // rax
  __int64 v17; // r13
  _BYTE *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  _BYTE *v22; // r14
  __int64 v23; // r12
  __int64 *v25; // rsi
  __int64 *v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rsi
  _BYTE *v30; // rbx
  __int64 *v31; // rdi
  __int64 *j; // r13
  void *v33; // [rsp+8h] [rbp-178h]
  unsigned __int64 v34; // [rsp+18h] [rbp-168h]
  void *v35; // [rsp+20h] [rbp-160h] BYREF
  __int64 *v36; // [rsp+28h] [rbp-158h]
  _BYTE *v37; // [rsp+40h] [rbp-140h] BYREF
  __int64 v38; // [rsp+48h] [rbp-138h]
  _BYTE v39[304]; // [rsp+50h] [rbp-130h] BYREF

  v6 = (_BYTE *)a1;
  v7 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v7 + 8) == 17 )
  {
    v8 = *(unsigned int *)(v7 + 32);
    v9 = v39;
    v38 = 0x2000000000LL;
    v10 = (__int64 *)v39;
    v11 = v39;
    v34 = v8;
    v12 = v8;
    v13 = v8;
    v37 = v39;
    if ( v8 )
    {
      if ( v8 > 0x20 )
      {
        sub_C8D5F0((__int64)&v37, v39, v8, 8u, (__int64)&v37, a6);
        v11 = v37;
        v9 = &v37[8 * (unsigned int)v38];
      }
      for ( i = &v11[8 * v34]; i != v9; ++v9 )
      {
        if ( v9 )
          *v9 = 0;
      }
      LODWORD(v38) = v12;
      v15 = 0;
      while ( 1 )
      {
        v17 = 8 * v15;
        v18 = (_BYTE *)sub_AD69F0(v6, (unsigned int)v15);
        v22 = v18;
        if ( !v18 )
          goto LABEL_11;
        if ( *v18 == 13 )
        {
          ++v15;
          *(_QWORD *)&v37[v17] = v18;
          if ( v34 == v15 )
          {
LABEL_16:
            v10 = (__int64 *)v37;
            v13 = (unsigned int)v38;
            break;
          }
        }
        else
        {
          if ( sub_AD8220((__int64)v18, (unsigned int)v15, v19, v20, v21) )
          {
            v25 = (__int64 *)(v22 + 24);
            v33 = sub_C33340();
            if ( *((void **)v22 + 3) == v33 )
              sub_C3C790(&v35, (_QWORD **)v25);
            else
              sub_C33EB0(&v35, v25);
            v26 = (__int64 *)&v35;
            if ( v35 == v33 )
              v26 = v36;
            sub_C39170((__int64)v26);
            v27 = sub_AD8F10(*((_QWORD *)v22 + 1), (__int64 *)&v35);
            *(_QWORD *)&v37[8 * v15] = v27;
            if ( v35 == v33 )
            {
              if ( v36 )
              {
                for ( j = &v36[3 * *(v36 - 1)]; v36 != j; sub_91D830(j) )
                  j -= 3;
                j_j_j___libc_free_0_0(j - 1);
              }
            }
            else
            {
              sub_C338F0((__int64)&v35);
            }
            goto LABEL_12;
          }
LABEL_11:
          v16 = sub_AD8F60(*(_QWORD *)(v7 + 24), 0, 0);
          *(_QWORD *)&v37[8 * v15] = v16;
LABEL_12:
          if ( v34 == ++v15 )
            goto LABEL_16;
        }
      }
    }
    v23 = sub_AD3730(v10, v13);
    if ( v37 != v39 )
      _libc_free(v37, v13);
    return (unsigned __int8 *)v23;
  }
  if ( sub_AD8220(a1, a2, a3, a4, a5) )
  {
    if ( *(_BYTE *)(v7 + 8) == 18 )
      v6 = sub_AD7630(a1, 0, v28);
    v29 = (__int64 *)(v6 + 24);
    v30 = sub_C33340();
    if ( *((_BYTE **)v6 + 3) == v30 )
      sub_C3C790(&v37, (_QWORD **)v29);
    else
      sub_C33EB0(&v37, v29);
    v31 = (__int64 *)&v37;
    if ( v37 == v30 )
      v31 = (__int64 *)v38;
    sub_C39170((__int64)v31);
    v23 = sub_AD8F10(v7, (__int64 *)&v37);
    sub_91D830(&v37);
    return (unsigned __int8 *)v23;
  }
  return sub_AD8F60(v7, 0, 0);
}
