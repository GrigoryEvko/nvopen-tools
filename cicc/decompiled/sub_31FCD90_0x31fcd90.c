// Function: sub_31FCD90
// Address: 0x31fcd90
//
void __fastcall sub_31FCD90(__int64 a1, _BYTE *a2, const __m128i *a3)
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
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  char *v16; // r9
  signed __int64 v17; // r10
  char *v18; // r14
  char *v19; // rax
  unsigned __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  _BYTE *v24; // [rsp+10h] [rbp-40h]
  signed __int64 v25; // [rsp+10h] [rbp-40h]
  char *v26; // [rsp+18h] [rbp-38h]
  _BYTE *v27; // [rsp+18h] [rbp-38h]
  char *v28; // [rsp+18h] [rbp-38h]
  _BYTE *v29; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((v4 - *(_QWORD *)a1) >> 4);
  if ( v6 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x5555555555555555LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4);
  v10 = a2 - v5;
  if ( v8 )
  {
    v20 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v11 = 0;
      v12 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x2AAAAAAAAAAAAAALL )
      v9 = 0x2AAAAAAAAAAAAAALL;
    v20 = 48 * v9;
  }
  v25 = a2 - v5;
  v29 = *(_BYTE **)a1;
  v21 = sub_22077B0(v20);
  v5 = v29;
  v10 = v25;
  v12 = (char *)v21;
  v11 = v21 + v20;
LABEL_7:
  v13 = (__m128i *)&v12[v10];
  if ( &v12[v10] )
  {
    v14 = _mm_loadu_si128(a3 + 1);
    v15 = _mm_loadu_si128(a3 + 2);
    *v13 = _mm_loadu_si128(a3);
    v13[1] = v14;
    v13[2] = v15;
  }
  v16 = &v12[v10 + 48];
  v17 = v4 - (_QWORD)a2;
  v18 = &v16[v4 - (_QWORD)a2];
  if ( v10 > 0 )
  {
    v22 = v17;
    v23 = (__int64)&v12[v10 + 48];
    v27 = v5;
    v19 = (char *)memmove(v12, v5, v10);
    v5 = v27;
    v17 = v22;
    v16 = (char *)v23;
    v12 = v19;
    if ( v22 <= 0 )
      goto LABEL_13;
LABEL_15:
    v24 = v5;
    v28 = v12;
    memcpy(v16, a2, v17);
    v5 = v24;
    v12 = v28;
    if ( !v24 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v17 > 0 )
    goto LABEL_15;
  if ( v5 )
  {
LABEL_13:
    v26 = v12;
    j_j___libc_free_0((unsigned __int64)v5);
    v12 = v26;
  }
LABEL_12:
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 8) = v18;
  *(_QWORD *)(a1 + 16) = v11;
}
