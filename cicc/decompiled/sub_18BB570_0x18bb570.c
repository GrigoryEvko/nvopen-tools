// Function: sub_18BB570
// Address: 0x18bb570
//
__int64 __fastcall sub_18BB570(char **a1, char *a2)
{
  char *v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // rdx
  __int64 v11; // r8
  char *v12; // rdx
  char *v13; // r15
  _QWORD *i; // r14
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  char *v25; // rdi
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  char *v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  _QWORD *v33; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  v31 = *a1;
  v4 = v3 - *a1;
  v5 = 0x6DB6DB6DB6DB6DB7LL * (v4 >> 4);
  if ( v5 == 0x124924924924924LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0x6DB6DB6DB6DB6DB7LL * (v4 >> 4);
  v8 = __CFADD__(v6, v5);
  v9 = v6 + v5;
  v10 = a2 - v31;
  if ( v8 )
  {
    v27 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v9 )
    {
      v29 = 0;
      v11 = 112;
      v33 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x124924924924924LL )
      v9 = 0x124924924924924LL;
    v27 = 112 * v9;
  }
  v30 = v27;
  v28 = sub_22077B0(v27);
  v10 = a2 - v31;
  v33 = (_QWORD *)v28;
  v11 = v28 + 112;
  v29 = v28 + v30;
LABEL_7:
  v12 = (char *)v33 + v10;
  if ( v12 )
    memset(v12, 0, 0x70u);
  v13 = v31;
  if ( a2 != v31 )
  {
    for ( i = v33; ; i += 14 )
    {
      if ( i )
      {
        *i = *(_QWORD *)v13;
        i[1] = *((_QWORD *)v13 + 1);
        i[2] = *((_QWORD *)v13 + 2);
        i[3] = *((_QWORD *)v13 + 3);
        i[4] = *((_QWORD *)v13 + 4);
        v15 = *((_QWORD *)v13 + 5);
        *((_QWORD *)v13 + 4) = 0;
        *((_QWORD *)v13 + 3) = 0;
        *((_QWORD *)v13 + 2) = 0;
        i[5] = v15;
        i[6] = *((_QWORD *)v13 + 6);
        i[7] = *((_QWORD *)v13 + 7);
        v16 = *((_QWORD *)v13 + 8);
        *((_QWORD *)v13 + 7) = 0;
        *((_QWORD *)v13 + 6) = 0;
        *((_QWORD *)v13 + 5) = 0;
        i[8] = v16;
        i[9] = *((_QWORD *)v13 + 9);
        i[10] = *((_QWORD *)v13 + 10);
        v17 = *((_QWORD *)v13 + 11);
        *((_QWORD *)v13 + 10) = 0;
        *((_QWORD *)v13 + 9) = 0;
        *((_QWORD *)v13 + 8) = 0;
        i[11] = v17;
        i[12] = *((_QWORD *)v13 + 12);
        i[13] = *((_QWORD *)v13 + 13);
        *((_QWORD *)v13 + 13) = 0;
        *((_QWORD *)v13 + 11) = 0;
      }
      else
      {
        v21 = *((_QWORD *)v13 + 11);
        if ( v21 )
          j_j___libc_free_0(v21, *((_QWORD *)v13 + 13) - v21);
      }
      v18 = *((_QWORD *)v13 + 8);
      if ( v18 )
        j_j___libc_free_0(v18, *((_QWORD *)v13 + 10) - v18);
      v19 = *((_QWORD *)v13 + 5);
      if ( v19 )
        j_j___libc_free_0(v19, *((_QWORD *)v13 + 7) - v19);
      v20 = *((_QWORD *)v13 + 2);
      if ( v20 )
        j_j___libc_free_0(v20, *((_QWORD *)v13 + 4) - v20);
      v13 += 112;
      if ( v13 == a2 )
        break;
    }
    v11 = (__int64)(i + 28);
  }
  if ( a2 != v3 )
  {
    v22 = a2;
    v23 = v11;
    do
    {
      v24 = *(_QWORD *)v22;
      v22 += 112;
      v23 += 112;
      *(_QWORD *)(v23 - 112) = v24;
      *(_QWORD *)(v23 - 104) = *((_QWORD *)v22 - 13);
      *(_QWORD *)(v23 - 96) = *((_QWORD *)v22 - 12);
      *(_QWORD *)(v23 - 88) = *((_QWORD *)v22 - 11);
      *(_QWORD *)(v23 - 80) = *((_QWORD *)v22 - 10);
      *(_QWORD *)(v23 - 72) = *((_QWORD *)v22 - 9);
      *(_QWORD *)(v23 - 64) = *((_QWORD *)v22 - 8);
      *(_QWORD *)(v23 - 56) = *((_QWORD *)v22 - 7);
      *(_QWORD *)(v23 - 48) = *((_QWORD *)v22 - 6);
      *(_QWORD *)(v23 - 40) = *((_QWORD *)v22 - 5);
      *(_QWORD *)(v23 - 32) = *((_QWORD *)v22 - 4);
      *(_QWORD *)(v23 - 24) = *((_QWORD *)v22 - 3);
      *(_QWORD *)(v23 - 16) = *((_QWORD *)v22 - 2);
      *(_QWORD *)(v23 - 8) = *((_QWORD *)v22 - 1);
    }
    while ( v22 != v3 );
    v11 += 112 * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)(v22 - a2 - 112) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 1);
  }
  v25 = v31;
  if ( v31 )
  {
    v32 = v11;
    j_j___libc_free_0(v25, a1[2] - v25);
    v11 = v32;
  }
  a1[1] = (char *)v11;
  *a1 = (char *)v33;
  a1[2] = (char *)v29;
  return v29;
}
