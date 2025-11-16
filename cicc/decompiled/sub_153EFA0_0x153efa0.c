// Function: sub_153EFA0
// Address: 0x153efa0
//
__int64 __fastcall sub_153EFA0(__int64 *a1, char *a2, _QWORD *a3, _QWORD *a4, unsigned __int64 *a5)
{
  __int64 v6; // rsi
  char *v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rbx
  bool v11; // cf
  unsigned __int64 v12; // rbx
  signed __int64 v13; // r10
  _QWORD *v14; // rbx
  __int64 v15; // rcx
  unsigned __int64 v16; // rax
  char *v17; // r15
  char *v18; // rax
  char *v19; // r15
  __int64 i; // rbx
  __int64 v21; // rdi
  __int64 v22; // rcx
  char *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  char *v26; // rdi
  __int64 result; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  unsigned __int64 *v30; // [rsp+8h] [rbp-68h]
  _QWORD *v31; // [rsp+10h] [rbp-60h]
  _QWORD *v32; // [rsp+18h] [rbp-58h]
  __int64 n; // [rsp+20h] [rbp-50h]
  __int64 v34; // [rsp+28h] [rbp-48h]
  char *v35; // [rsp+30h] [rbp-40h]
  __int64 v36; // [rsp+30h] [rbp-40h]
  __int64 v37; // [rsp+38h] [rbp-38h]

  v6 = 0x333333333333333LL;
  v8 = (char *)a1[1];
  v35 = (char *)*a1;
  v9 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v8[-*a1] >> 3);
  if ( v9 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * ((a1[1] - *a1) >> 3);
  v11 = __CFADD__(v9, v10);
  v12 = v9 + v10;
  v34 = v12;
  v13 = a2 - v35;
  if ( v11 )
  {
    v28 = 0x7FFFFFFFFFFFFFF8LL;
    v34 = 0x333333333333333LL;
  }
  else
  {
    if ( !v12 )
    {
      v37 = 0;
      goto LABEL_7;
    }
    if ( v12 <= 0x333333333333333LL )
      v6 = v12;
    v34 = v6;
    v28 = 40 * v6;
  }
  v30 = a5;
  v31 = a4;
  v32 = a3;
  v29 = sub_22077B0(v28);
  v13 = a2 - v35;
  a3 = v32;
  a4 = v31;
  a5 = v30;
  v37 = v29;
LABEL_7:
  v14 = (_QWORD *)(v37 + v13);
  if ( v37 + v13 )
  {
    v15 = *a4;
    v16 = *a5;
    *v14 = *a3;
    v14[1] = v15;
    if ( v16 > 0x1FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v14[2] = 0;
    v17 = 0;
    v14[3] = 0;
    v14[4] = 0;
    if ( v16 )
    {
      n = 4 * v16;
      v18 = (char *)sub_22077B0(4 * v16);
      v14[2] = v18;
      v17 = &v18[n];
      v14[4] = &v18[n];
      if ( v18 != &v18[n] )
        memset(v18, 0, n);
    }
    v14[3] = v17;
  }
  v19 = v35;
  for ( i = v37; v19 != a2; i = 40 )
  {
    while ( i )
    {
      *(_QWORD *)i = *(_QWORD *)v19;
      *(_QWORD *)(i + 8) = *((_QWORD *)v19 + 1);
      *(_QWORD *)(i + 16) = *((_QWORD *)v19 + 2);
      *(_QWORD *)(i + 24) = *((_QWORD *)v19 + 3);
      *(_QWORD *)(i + 32) = *((_QWORD *)v19 + 4);
      *((_QWORD *)v19 + 4) = 0;
      *((_QWORD *)v19 + 2) = 0;
LABEL_16:
      v19 += 40;
      i += 40;
      if ( v19 == a2 )
        goto LABEL_20;
    }
    v21 = *((_QWORD *)v19 + 2);
    if ( !v21 )
      goto LABEL_16;
    j_j___libc_free_0(v21, *((_QWORD *)v19 + 4) - v21);
    v19 += 40;
  }
LABEL_20:
  v22 = i + 40;
  if ( a2 != v8 )
  {
    v23 = a2;
    v24 = i + 40;
    do
    {
      v25 = *(_QWORD *)v23;
      v23 += 40;
      v24 += 40;
      *(_QWORD *)(v24 - 40) = v25;
      *(_QWORD *)(v24 - 32) = *((_QWORD *)v23 - 4);
      *(_QWORD *)(v24 - 24) = *((_QWORD *)v23 - 3);
      *(_QWORD *)(v24 - 16) = *((_QWORD *)v23 - 2);
      *(_QWORD *)(v24 - 8) = *((_QWORD *)v23 - 1);
    }
    while ( v23 != v8 );
    v22 += 8 * ((unsigned __int64)(v23 - a2 - 40) >> 3) + 40;
  }
  v26 = v35;
  if ( v35 )
  {
    v36 = v22;
    j_j___libc_free_0(v26, a1[2] - (_QWORD)v26);
    v22 = v36;
  }
  a1[1] = v22;
  *a1 = v37;
  result = v37 + 40 * v34;
  a1[2] = result;
  return result;
}
