// Function: sub_2864680
// Address: 0x2864680
//
void __fastcall sub_2864680(const __m128i **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v8; // rbx
  char v9; // r12
  __m128i v10; // xmm0
  const __m128i *v11; // rdx
  __int8 v12; // al
  __int64 v13; // rax
  __m128i v14; // xmm1
  __int64 v15; // rax
  __int8 v16; // dl
  __int64 v17; // rax
  bool v18; // zf
  char v19; // dl
  const __m128i *v20; // rdi
  const __m128i *v21; // rax
  __int64 *v22; // rdi
  unsigned int v23; // r9d
  char v24; // r8
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // rdx
  __int64 *v28; // rdi
  _QWORD *v29; // r12
  __int64 *v30; // rdx
  __int64 *v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  const __m128i *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rax
  int v37; // eax
  __int64 v38; // rax
  __int64 *v39; // [rsp+8h] [rbp-E8h]
  __int64 v40; // [rsp+10h] [rbp-E0h]
  __int64 *v41; // [rsp+10h] [rbp-E0h]
  __int64 *v42; // [rsp+10h] [rbp-E0h]
  __int64 **v43; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+38h] [rbp-B8h]
  __int64 *v45; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD *v46; // [rsp+48h] [rbp-A8h]
  __int64 v47; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v48; // [rsp+58h] [rbp-98h]
  __int8 v49; // [rsp+68h] [rbp-88h]
  __int64 v50; // [rsp+70h] [rbp-80h]
  _BYTE *v51; // [rsp+78h] [rbp-78h] BYREF
  __int64 v52; // [rsp+80h] [rbp-70h]
  _BYTE v53[32]; // [rsp+88h] [rbp-68h] BYREF
  _QWORD *v54; // [rsp+A8h] [rbp-48h]
  __m128i v55; // [rsp+B0h] [rbp-40h]

  v8 = *a1;
  v9 = a4;
  v10 = _mm_loadu_si128((const __m128i *)&(*a1)->m128i_u64[1]);
  v11 = v8;
  v47 = (*a1)->m128i_i64[0];
  v12 = v8[1].m128i_i8[8];
  v48 = v10;
  v49 = v12;
  v13 = v8[2].m128i_i64[0];
  v51 = v53;
  v50 = v13;
  v52 = 0x400000000LL;
  if ( v8[3].m128i_i32[0] )
  {
    sub_2850210((__int64)&v51, (__int64)&v8[2].m128i_i64[1], (__int64)v8, a4, a5, a6);
    v11 = *a1;
  }
  v14 = _mm_loadu_si128(v8 + 6);
  v54 = (_QWORD *)v8[5].m128i_i64[1];
  v55 = v14;
  v15 = v11->m128i_i64[1];
  v16 = v11[1].m128i_i8[0];
  if ( !v15 || !a3 || v9 == v16 )
  {
    v17 = v15 - a3;
    v18 = v16 == 0;
    v19 = 1;
    v20 = a1[1];
    if ( v18 )
      v19 = v9;
    v48.m128i_i64[0] = v17;
    v21 = a1[2];
    v22 = (__int64 *)v20[3].m128i_i64[0];
    v23 = v21[2].m128i_u32[0];
    v24 = v21[46].m128i_i8[0];
    v48.m128i_i8[8] = v19;
    if ( sub_2850770(
           v22,
           v21[44].m128i_i64[1],
           v21[45].m128i_i8[0],
           v21[45].m128i_i64[1],
           v24,
           v23,
           v21[2].m128i_i64[1],
           v21[3].m128i_u32[0],
           (__int64)&v47) )
    {
      v25 = sub_D95540(a2);
      v40 = a1[1]->m128i_i64[1];
      v26 = sub_DA2C50(v40, v25, a3, 0);
      v27 = v26;
      if ( v9 )
      {
        v39 = (__int64 *)v40;
        v41 = v26;
        v35 = sub_D95540((__int64)v26);
        v46 = sub_DA3710((__int64)v39, v35);
        v45 = v41;
        v43 = &v45;
        v44 = 0x200000002LL;
        v36 = sub_DC8BD0(v39, (__int64)&v43, 0, 0);
        v27 = v36;
        if ( v43 != &v45 )
        {
          v42 = v36;
          _libc_free((unsigned __int64)v43);
          v27 = v42;
        }
      }
      v28 = (__int64 *)a1[1]->m128i_i64[1];
      v45 = v27;
      v43 = &v45;
      v46 = (_QWORD *)a2;
      v44 = 0x200000002LL;
      v29 = sub_DC7EB0(v28, (__int64)&v43, 0, 0);
      if ( v43 != &v45 )
        _libc_free((unsigned __int64)v43);
      v18 = !sub_D968A0((__int64)v29);
      v34 = a1[3];
      if ( v18 )
      {
        if ( v34->m128i_i8[0] )
          v54 = v29;
        else
          *(_QWORD *)&v51[8 * a1[4]->m128i_i64[0]] = v29;
      }
      else
      {
        if ( v34->m128i_i8[0] )
        {
          v50 = 0;
          v54 = 0;
        }
        else
        {
          v31 = (__int64 *)&v51[8 * a1[4]->m128i_i64[0]];
          v30 = (__int64 *)&v51[8 * (unsigned int)v52 - 8];
          v37 = v52;
          if ( v31 != v30 )
          {
            v38 = *v31;
            *v31 = *v30;
            *v30 = v38;
            v37 = v52;
          }
          LODWORD(v52) = v37 - 1;
        }
        sub_2857080((__int64)&v47, a1[1][3].m128i_i64[1], (__int64)v30, (__int64)v31, v32, v33);
      }
      sub_2862B30((__int64)a1[1], (__int64)a1[2], a1[5]->m128i_i32[0], (unsigned __int64)&v47, v32, v33);
    }
  }
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
}
