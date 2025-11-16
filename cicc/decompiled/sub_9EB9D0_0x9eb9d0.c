// Function: sub_9EB9D0
// Address: 0x9eb9d0
//
__int64 __fastcall sub_9EB9D0(__int64 a1, char *a2, __int64 a3)
{
  __int64 v3; // rcx
  char *v5; // r13
  char *v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  signed __int64 v12; // rdx
  _QWORD *v13; // rax
  bool v14; // zf
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  char *v21; // rbx
  char *v22; // rax
  char *v23; // rsi
  const void **v24; // rcx
  __int64 *v25; // r12
  __int64 *v26; // r14
  __int64 v27; // rdi
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // rdi
  __int64 v32; // rcx
  __int64 v33; // rax
  char *i; // [rsp+10h] [rbp-50h]
  _QWORD *v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  size_t v39; // [rsp+28h] [rbp-38h]

  v3 = 0x124924924924924LL;
  v5 = *(char **)(a1 + 8);
  v6 = *(char **)a1;
  v7 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)&v5[-*(_QWORD *)a1] >> 4);
  if ( v7 == 0x124924924924924LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0x6DB6DB6DB6DB6DB7LL * ((v5 - v6) >> 4);
  v10 = __CFADD__(v8, v7);
  v11 = v8 + v7;
  v12 = a2 - v6;
  if ( v10 )
  {
    v32 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v11 )
    {
      v36 = 0;
      v39 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x124924924924924LL )
      v11 = 0x124924924924924LL;
    v32 = 112 * v11;
  }
  v37 = v32;
  v33 = sub_22077B0(v32);
  v12 = a2 - v6;
  v39 = v33;
  v3 = v33 + v37;
  v36 = v33 + v37;
LABEL_7:
  v13 = (_QWORD *)(v39 + v12);
  if ( v39 + v12 )
  {
    v14 = *(_QWORD *)(a3 + 8) == 0;
    v13[1] = 0;
    *v13 = v13 + 3;
    v13[2] = 40;
    if ( !v14 )
    {
      v35 = (_QWORD *)(v39 + v12);
      sub_9C3060((__int64)v13, (char **)a3);
      v13 = v35;
    }
    v15 = *(_QWORD *)(a3 + 64);
    *(_QWORD *)(a3 + 64) = 0;
    v13[8] = v15;
    v16 = *(_QWORD *)(a3 + 72);
    *(_QWORD *)(a3 + 72) = 0;
    v13[9] = v16;
    v17 = *(_QWORD *)(a3 + 80);
    *(_QWORD *)(a3 + 80) = 0;
    v13[10] = v17;
    v18 = *(_QWORD *)(a3 + 88);
    *(_QWORD *)(a3 + 88) = 0;
    v13[11] = v18;
    v19 = *(_QWORD *)(a3 + 96);
    *(_QWORD *)(a3 + 96) = 0;
    v13[12] = v19;
    v20 = *(_QWORD *)(a3 + 104);
    *(_QWORD *)(a3 + 104) = 0;
    v13[13] = v20;
  }
  v21 = v6;
  v22 = sub_9EB710(v6, a2, v39, (const void **)v3);
  v23 = v5;
  for ( i = sub_9EB710(a2, v5, (size_t)(v22 + 112), v24); v21 != v5; v21 += 112 )
  {
    v25 = (__int64 *)*((_QWORD *)v21 + 12);
    v26 = (__int64 *)*((_QWORD *)v21 + 11);
    if ( v25 != v26 )
    {
      do
      {
        v27 = *v26;
        if ( *v26 )
        {
          v23 = (char *)(v26[2] - v27);
          j_j___libc_free_0(v27, v23);
        }
        v26 += 3;
      }
      while ( v25 != v26 );
      v26 = (__int64 *)*((_QWORD *)v21 + 11);
    }
    if ( v26 )
    {
      v23 = (char *)(*((_QWORD *)v21 + 13) - (_QWORD)v26);
      j_j___libc_free_0(v26, v23);
    }
    v28 = *((_QWORD *)v21 + 9);
    v29 = *((_QWORD *)v21 + 8);
    if ( v28 != v29 )
    {
      do
      {
        v30 = *(_QWORD *)(v29 + 8);
        if ( v30 != v29 + 24 )
          _libc_free(v30, v23);
        v29 += 72;
      }
      while ( v28 != v29 );
      v29 = *((_QWORD *)v21 + 8);
    }
    if ( v29 )
    {
      v23 = (char *)(*((_QWORD *)v21 + 10) - v29);
      j_j___libc_free_0(v29, v23);
    }
    if ( *(char **)v21 != v21 + 24 )
      _libc_free(*(_QWORD *)v21, v23);
  }
  if ( v6 )
    j_j___libc_free_0(v6, *(_QWORD *)(a1 + 16) - (_QWORD)v6);
  *(_QWORD *)a1 = v39;
  *(_QWORD *)(a1 + 8) = i;
  *(_QWORD *)(a1 + 16) = v36;
  return a1;
}
