// Function: sub_26BA1A0
// Address: 0x26ba1a0
//
void __fastcall sub_26BA1A0(__int64 a1, _BYTE *a2, const __m128i *a3)
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
  __m128i *v14; // rax
  __m128i v15; // xmm1
  char *v16; // r9
  signed __int64 v17; // r10
  char *v18; // r14
  char *v19; // rax
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  _BYTE *v24; // [rsp+10h] [rbp-40h]
  signed __int64 v25; // [rsp+10h] [rbp-40h]
  char *v26; // [rsp+18h] [rbp-38h]
  _BYTE *v27; // [rsp+18h] [rbp-38h]
  char *v28; // [rsp+18h] [rbp-38h]
  _BYTE *v29; // [rsp+18h] [rbp-38h]

  v3 = 0x3FFFFFFFFFFFFFFLL;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = (v4 - *(_QWORD *)a1) >> 5;
  if ( v6 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v6 )
    v8 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 5;
  v9 = __CFADD__(v8, v6);
  v10 = v8 + v6;
  v11 = (char *)v9;
  v12 = a2 - v5;
  if ( v9 )
  {
    v20 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 <= 0x3FFFFFFFFFFFFFFLL )
      v3 = v10;
    v20 = 32 * v3;
  }
  v25 = a2 - v5;
  v29 = *(_BYTE **)a1;
  v21 = sub_22077B0(v20);
  v5 = v29;
  v12 = v25;
  v11 = (char *)v21;
  v13 = v21 + v20;
LABEL_7:
  v14 = (__m128i *)&v11[v12];
  if ( &v11[v12] )
  {
    v15 = _mm_loadu_si128(a3 + 1);
    *v14 = _mm_loadu_si128(a3);
    v14[1] = v15;
  }
  v16 = &v11[v12 + 32];
  v17 = v4 - (_QWORD)a2;
  v18 = &v16[v4 - (_QWORD)a2];
  if ( v12 > 0 )
  {
    v22 = v17;
    v23 = (__int64)&v11[v12 + 32];
    v27 = v5;
    v19 = (char *)memmove(v11, v5, v12);
    v5 = v27;
    v17 = v22;
    v16 = (char *)v23;
    v11 = v19;
    if ( v22 <= 0 )
      goto LABEL_13;
LABEL_15:
    v24 = v5;
    v28 = v11;
    memcpy(v16, a2, v17);
    v5 = v24;
    v11 = v28;
    if ( !v24 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v17 > 0 )
    goto LABEL_15;
  if ( v5 )
  {
LABEL_13:
    v26 = v11;
    j_j___libc_free_0((unsigned __int64)v5);
    v11 = v26;
  }
LABEL_12:
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = v18;
  *(_QWORD *)(a1 + 16) = v13;
}
