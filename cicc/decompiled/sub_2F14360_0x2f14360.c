// Function: sub_2F14360
// Address: 0x2f14360
//
void __fastcall sub_2F14360(__int64 a1, _BYTE *a2, const __m128i *a3)
{
  __int64 v4; // r14
  _BYTE *v5; // r8
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // rdx
  unsigned __int64 v11; // r13
  char *v12; // rcx
  __m128i *v13; // rax
  __m128i v14; // xmm0
  char *v15; // r9
  signed __int64 v16; // r10
  char *v17; // r14
  char *v18; // rax
  unsigned __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  _BYTE *v23; // [rsp+10h] [rbp-40h]
  signed __int64 v24; // [rsp+10h] [rbp-40h]
  char *v25; // [rsp+18h] [rbp-38h]
  _BYTE *v26; // [rsp+18h] [rbp-38h]
  char *v27; // [rsp+18h] [rbp-38h]
  _BYTE *v28; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((v4 - *(_QWORD *)a1) >> 2);
  if ( v6 == 0x666666666666666LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 2);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x3333333333333333LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 2);
  v10 = a2 - v5;
  if ( v8 )
  {
    v19 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v11 = 0;
      v12 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x666666666666666LL )
      v9 = 0x666666666666666LL;
    v19 = 20 * v9;
  }
  v24 = a2 - v5;
  v28 = *(_BYTE **)a1;
  v20 = sub_22077B0(v19);
  v5 = v28;
  v10 = v24;
  v12 = (char *)v20;
  v11 = v20 + v19;
LABEL_7:
  v13 = (__m128i *)&v12[v10];
  if ( &v12[v10] )
  {
    v14 = _mm_loadu_si128(a3);
    v13[1].m128i_i32[0] = a3[1].m128i_i32[0];
    *v13 = v14;
  }
  v15 = &v12[v10 + 20];
  v16 = v4 - (_QWORD)a2;
  v17 = &v15[v4 - (_QWORD)a2];
  if ( v10 > 0 )
  {
    v21 = v16;
    v22 = (__int64)&v12[v10 + 20];
    v26 = v5;
    v18 = (char *)memmove(v12, v5, v10);
    v5 = v26;
    v16 = v21;
    v15 = (char *)v22;
    v12 = v18;
    if ( v21 <= 0 )
      goto LABEL_13;
LABEL_15:
    v23 = v5;
    v27 = v12;
    memcpy(v15, a2, v16);
    v5 = v23;
    v12 = v27;
    if ( !v23 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v16 > 0 )
    goto LABEL_15;
  if ( v5 )
  {
LABEL_13:
    v25 = v12;
    j_j___libc_free_0((unsigned __int64)v5);
    v12 = v25;
  }
LABEL_12:
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 8) = v17;
  *(_QWORD *)(a1 + 16) = v11;
}
