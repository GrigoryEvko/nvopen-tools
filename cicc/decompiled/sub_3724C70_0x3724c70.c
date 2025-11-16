// Function: sub_3724C70
// Address: 0x3724c70
//
void __fastcall sub_3724C70(__int64 a1, const void *a2, _QWORD *a3, char *a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  unsigned __int64 v8; // r15
  __int64 *v9; // r12
  __int64 *v10; // r10
  __int64 v11; // rcx
  void *v12; // rdx
  char *v13; // rax
  unsigned __int64 v14; // r15
  __int64 v15; // rdi
  char *v16; // rax
  __int64 v17; // r8
  unsigned __int64 *v18; // r15
  char **v19; // rsi
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rax
  __int64 v22; // r13
  __int64 j; // r14
  _BYTE *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdi
  char *v27; // rax
  char **v28; // rax
  char **v29; // rbx
  char *v30; // r14
  char *v31; // r15
  __int64 v32; // rcx
  void *v33; // rdx
  char *v34; // rax
  unsigned __int64 v35; // r12
  unsigned __int64 *v36; // r12
  unsigned __int64 *v37; // r13
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+18h] [rbp-88h]
  __int64 i; // [rsp+20h] [rbp-80h]
  __int64 *v42; // [rsp+28h] [rbp-78h]
  char **v43; // [rsp+28h] [rbp-78h]
  __int64 v44; // [rsp+30h] [rbp-70h]
  __int64 v45; // [rsp+30h] [rbp-70h]
  void *v46; // [rsp+38h] [rbp-68h]
  void *v47; // [rsp+38h] [rbp-68h]
  __int64 v48[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v49; // [rsp+60h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 128);
  v40 = (__int64)a2;
  v39 = (__int64)a3;
  v8 = (unsigned __int64)*(unsigned int *)(a1 + 136) << 6;
  v38 = (__int64)a4;
  for ( i = v7 + v8; v7 != i; v7 += 64 )
  {
LABEL_2:
    v9 = *(__int64 **)(v7 + 40);
    v10 = *(__int64 **)(v7 + 32);
    v11 = v9 - v10;
    if ( (char *)v9 - (char *)v10 <= 0 )
    {
LABEL_36:
      v14 = 0;
      sub_37230A0(v10, v9);
    }
    else
    {
      v12 = &unk_435FF63;
      while ( 1 )
      {
        v42 = v10;
        v44 = v11;
        v46 = v12;
        v13 = (char *)sub_2207800(8 * v11);
        v12 = v46;
        v10 = v42;
        v14 = (unsigned __int64)v13;
        if ( v13 )
          break;
        v11 = v44 >> 1;
        if ( !(v44 >> 1) )
          goto LABEL_36;
      }
      sub_3724720(v42, v9, v13, v44);
    }
    j_j___libc_free_0(v14);
    a6 = *(_QWORD *)(v7 + 40);
    a4 = *(char **)(v7 + 32);
    a2 = (const void *)a6;
    if ( (char *)a6 != a4 )
    {
      while ( 1 )
      {
        v16 = a4;
        a4 += 8;
        if ( (char *)a6 == a4 )
          break;
        v15 = *((_QWORD *)a4 - 1);
        if ( v15 == *((_QWORD *)v16 + 1) )
        {
          if ( (char *)a6 == v16 )
            break;
          a3 = v16 + 16;
          if ( (char *)a6 != v16 + 16 )
          {
            while ( 1 )
            {
              if ( *a3 != v15 )
              {
                *((_QWORD *)v16 + 1) = *a3;
                v16 += 8;
              }
              if ( (_QWORD *)a6 == ++a3 )
                break;
              v15 = *(_QWORD *)v16;
            }
            a4 = v16 + 8;
            if ( (char *)a6 == v16 + 8 )
              break;
            a6 = *(_QWORD *)(v7 + 40);
          }
          if ( a2 != (const void *)a6 )
          {
            v27 = (char *)memmove(a4, a2, a6 - (_QWORD)a2);
            a6 = *(_QWORD *)(v7 + 40);
            a4 = v27;
          }
          a4 += a6 - (_QWORD)a2;
          if ( a4 == (char *)a6 )
            break;
          *(_QWORD *)(v7 + 40) = a4;
          v7 += 64;
          if ( v7 != i )
            goto LABEL_2;
          goto LABEL_11;
        }
      }
    }
  }
LABEL_11:
  sub_37234B0(a1, (__int64)a2, (__int64)a3, (__int64)a4, a5, a6);
  v18 = *(unsigned __int64 **)(a1 + 192);
  v19 = *(char ***)(a1 + 184);
  v20 = *(unsigned int *)(a1 + 152);
  v21 = 0xAAAAAAAAAAAAAAABLL * (((char *)v18 - (char *)v19) >> 3);
  if ( v20 > v21 )
  {
    sub_3723B20(a1 + 184, v20 - v21);
    v19 = *(char ***)(a1 + 184);
  }
  else if ( v20 < v21 )
  {
    v36 = (unsigned __int64 *)&v19[3 * v20];
    if ( v18 != v36 )
    {
      v37 = (unsigned __int64 *)&v19[3 * v20];
      do
      {
        if ( *v37 )
          j_j___libc_free_0(*v37);
        v37 += 3;
      }
      while ( v18 != v37 );
      *(_QWORD *)(a1 + 192) = v36;
      v19 = *(char ***)(a1 + 184);
    }
  }
  v22 = *(_QWORD *)(a1 + 128);
  for ( j = v22 + ((unsigned __int64)*(unsigned int *)(a1 + 136) << 6); v22 != j; v19 = *(char ***)(a1 + 184) )
  {
    v25 = (unsigned int)(*(_DWORD *)(v22 + 24) % *(_DWORD *)(a1 + 152));
    v26 = (__int64)&v19[3 * v25];
    v48[0] = v22 + 16;
    v24 = *(_BYTE **)(v26 + 8);
    if ( v24 == *(_BYTE **)(v26 + 16) )
    {
      sub_3723D60(v26, v24, v48);
    }
    else
    {
      if ( v24 )
      {
        *(_QWORD *)v24 = v22 + 16;
        v24 = *(_BYTE **)(v26 + 8);
      }
      *(_QWORD *)(v26 + 8) = v24 + 8;
    }
    v22 += 64;
    v49 = 261;
    v48[0] = v39;
    v48[1] = v38;
    *(_QWORD *)(v22 - 8) = sub_31DCC50(v40, v48, v25, v20, v17);
  }
  v28 = *(char ***)(a1 + 192);
  v29 = v19;
  v43 = v28;
  while ( v43 != v29 )
  {
    v30 = v29[1];
    v31 = *v29;
    v32 = (v30 - *v29) >> 3;
    if ( v30 - *v29 <= 0 )
    {
LABEL_45:
      v35 = 0;
      sub_3723430(v31, v30);
    }
    else
    {
      v33 = &unk_435FF63;
      while ( 1 )
      {
        v45 = v32;
        v47 = v33;
        v34 = (char *)sub_2207800(8 * v32);
        v33 = v47;
        v35 = (unsigned __int64)v34;
        if ( v34 )
          break;
        v32 = v45 >> 1;
        if ( !(v45 >> 1) )
          goto LABEL_45;
      }
      sub_3724BA0(v31, v30, v34, (char *)v45);
    }
    v29 += 3;
    j_j___libc_free_0(v35);
  }
}
