// Function: sub_28CDB80
// Address: 0x28cdb80
//
unsigned __int64 __fastcall sub_28CDB80(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r8
  _QWORD *v4; // rax
  __int64 v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __m128i *v8; // r15
  char *v9; // r14
  char *v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  char *v17; // r8
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __m128i v21; // [rsp+10h] [rbp-110h] BYREF
  __m128i v22; // [rsp+20h] [rbp-100h] BYREF
  __m128i v23; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v24; // [rsp+40h] [rbp-E0h]
  __int64 v25; // [rsp+58h] [rbp-C8h] BYREF
  __int64 src; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v27; // [rsp+68h] [rbp-B8h] BYREF
  _OWORD dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v29; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v30; // [rsp+C0h] [rbp-60h]
  __m128i v31; // [rsp+D0h] [rbp-50h]
  __int64 v32; // [rsp+E0h] [rbp-40h]
  void (__fastcall *v33)(__int64, __int64); // [rsp+E8h] [rbp-38h]

  v3 = *a1;
  v29 = 0u;
  v30 = 0u;
  v31 = 0u;
  v32 = 0;
  v33 = sub_C64CA0;
  v25 = 0;
  memset(dest, 0, sizeof(dest));
  v4 = sub_CA5190((unsigned __int64 *)dest, &v25, dest, (unsigned __int64)&v29, v3);
  v5 = v25;
  v6 = v4;
  v7 = *a2;
  v8 = (__m128i *)(v6 + 1);
  src = *a2;
  if ( v6 + 1 <= (_QWORD *)&v29 )
  {
    *v6 = v7;
  }
  else
  {
    v9 = (char *)((char *)&v29 - (char *)v6);
    memcpy(v6, &src, (char *)&v29 - (char *)v6);
    if ( v5 )
    {
      v5 += 64;
      sub_AC2A10((unsigned __int64 *)&v29, dest);
    }
    else
    {
      v5 = 64;
      sub_AC28A0((unsigned __int64 *)&v21, (__int64 *)dest, (unsigned __int64)v33);
      v18 = _mm_loadu_si128(&v22);
      v19 = _mm_loadu_si128(&v23);
      v29 = _mm_loadu_si128(&v21);
      v32 = v24;
      v30 = v18;
      v31 = v19;
    }
    v8 = (__m128i *)((char *)dest + 8LL - (_QWORD)v9);
    if ( v8 > &v29 )
LABEL_5:
      BUG();
    memcpy(dest, (char *)&src + (_QWORD)v9, 8LL - (_QWORD)v9);
  }
  v10 = &v8->m128i_i8[8];
  v11 = *a3;
  v27 = *a3;
  if ( &v8->m128i_u64[1] > (unsigned __int64 *)&v29 )
  {
    memcpy(v8, &v27, (char *)&v29 - (char *)v8);
    if ( v5 )
    {
      v5 += 64;
      sub_AC2A10((unsigned __int64 *)&v29, dest);
      v17 = (char *)((char *)&v29 - (char *)v8);
    }
    else
    {
      v5 = 64;
      sub_AC28A0((unsigned __int64 *)&v21, (__int64 *)dest, (unsigned __int64)v33);
      v14 = _mm_loadu_si128(&v21);
      v15 = _mm_loadu_si128(&v22);
      v16 = _mm_loadu_si128(&v23);
      v32 = v24;
      v17 = (char *)((char *)&v29 - (char *)v8);
      v29 = v14;
      v30 = v15;
      v31 = v16;
    }
    v10 = (char *)dest + 8LL - (_QWORD)v17;
    if ( v10 > (char *)&v29 )
      goto LABEL_5;
    v12 = 8LL - (_QWORD)v17;
    memcpy(dest, (char *)&v27 + (_QWORD)v17, 8LL - (_QWORD)v17);
    if ( !v5 )
      return sub_AC25F0(dest, v12, (__int64)v33);
  }
  else
  {
    v8->m128i_i64[0] = v11;
    v12 = v10 - (char *)dest;
    if ( !v5 )
      return sub_AC25F0(dest, v12, (__int64)v33);
  }
  sub_28C7830((char *)dest, v10, v29.m128i_i8);
  sub_AC2A10((unsigned __int64 *)&v29, dest);
  return sub_AF1490(&v29, v12 + v5);
}
