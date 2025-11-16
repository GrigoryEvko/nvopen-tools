// Function: sub_39F2800
// Address: 0x39f2800
//
void __fastcall sub_39F2800(__int64 a1, _BYTE *a2, const __m128i *a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  _BYTE *v5; // r8
  __int64 v6; // rax
  __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rcx
  signed __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  char *v14; // r9
  signed __int64 v15; // r10
  char *v16; // r14
  char *v17; // rax
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  _BYTE *v22; // [rsp+10h] [rbp-40h]
  signed __int64 v23; // [rsp+10h] [rbp-40h]
  char *v24; // [rsp+18h] [rbp-38h]
  _BYTE *v25; // [rsp+18h] [rbp-38h]
  char *v26; // [rsp+18h] [rbp-38h]
  _BYTE *v27; // [rsp+18h] [rbp-38h]

  v3 = 0x7FFFFFFFFFFFFFFLL;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = (v4 - *(_QWORD *)a1) >> 4;
  if ( v6 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v6 )
    v8 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4;
  v9 = __CFADD__(v8, v6);
  v10 = v8 + v6;
  v11 = (char *)v9;
  v12 = a2 - v5;
  if ( v9 )
  {
    v18 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v10 )
    {
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 <= 0x7FFFFFFFFFFFFFFLL )
      v3 = v10;
    v18 = 16 * v3;
  }
  v23 = a2 - v5;
  v27 = *(_BYTE **)a1;
  v19 = sub_22077B0(v18);
  v5 = v27;
  v12 = v23;
  v11 = (char *)v19;
  v13 = v19 + v18;
LABEL_7:
  if ( &v11[v12] )
    *(__m128i *)&v11[v12] = _mm_loadu_si128(a3);
  v14 = &v11[v12 + 16];
  v15 = v4 - (_QWORD)a2;
  v16 = &v14[v4 - (_QWORD)a2];
  if ( v12 > 0 )
  {
    v20 = v15;
    v21 = (__int64)&v11[v12 + 16];
    v25 = v5;
    v17 = (char *)memmove(v11, v5, v12);
    v5 = v25;
    v15 = v20;
    v14 = (char *)v21;
    v11 = v17;
    if ( v20 <= 0 )
      goto LABEL_13;
LABEL_15:
    v22 = v5;
    v26 = v11;
    memcpy(v14, a2, v15);
    v5 = v22;
    v11 = v26;
    if ( !v22 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v15 > 0 )
    goto LABEL_15;
  if ( v5 )
  {
LABEL_13:
    v24 = v11;
    j_j___libc_free_0((unsigned __int64)v5);
    v11 = v24;
  }
LABEL_12:
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = v16;
  *(_QWORD *)(a1 + 16) = v13;
}
