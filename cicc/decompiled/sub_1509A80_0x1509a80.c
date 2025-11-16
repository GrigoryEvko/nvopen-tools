// Function: sub_1509A80
// Address: 0x1509a80
//
__int64 __fastcall sub_1509A80(
        __int64 a1,
        __m128i **a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  unsigned int v7; // r14d
  unsigned int v8; // r13d
  __m128i *v12; // rdi
  unsigned __int64 v13; // rax
  __int64 (*v14)(void); // rax
  char v15; // dl
  __int64 v17; // rdi
  __m128i *v18; // rax
  __int64 v19; // rdx
  __m128i *v20; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v21; // [rsp+18h] [rbp-38h]
  char *v22; // [rsp+20h] [rbp-30h]
  __int64 v23; // [rsp+28h] [rbp-28h]

  v7 = (unsigned __int8)a5;
  v8 = a4;
  v12 = *a2;
  v13 = (*a2)[1].m128i_i64[0] - (*a2)->m128i_i64[1];
  v20 = (__m128i *)(*a2)->m128i_i64[1];
  v21 = v13;
  v14 = *(__int64 (**)(void))(v12->m128i_i64[0] + 16);
  if ( (char *)v14 == (char *)sub_12BCB10 )
  {
    v23 = 14;
    v22 = "Unknown buffer";
  }
  else
  {
    v22 = (char *)v14();
    v23 = v19;
  }
  sub_15099C0(a1, a3, v8, v7, a5, a6, a7, v20, v21);
  v15 = *(_BYTE *)(a1 + 8) & 1;
  *(_BYTE *)(a1 + 8) = (2 * v15) | *(_BYTE *)(a1 + 8) & 0xFD;
  if ( v15 )
    return a1;
  v17 = *(_QWORD *)a1;
  v18 = *a2;
  *a2 = 0;
  v20 = v18;
  sub_1633DB0(v17, &v20);
  if ( !v20 )
    return a1;
  (*(void (__fastcall **)(__m128i *))(v20->m128i_i64[0] + 8))(v20);
  return a1;
}
