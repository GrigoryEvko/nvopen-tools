// Function: sub_19E8720
// Address: 0x19e8720
//
unsigned __int64 __fastcall sub_19E8720(__int64 a1)
{
  __int64 *v1; // r13
  unsigned __int64 v3; // r15
  __int8 *v4; // rax
  __int64 v5; // rcx
  __int8 *v6; // rdi
  __int64 v7; // rax
  char *v8; // r9
  char *v9; // r8
  __int64 v10; // rcx
  __int8 *v11; // rax
  __int64 v12; // r15
  unsigned __int64 v13; // rax
  signed __int64 v15; // rbx
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // [rsp+0h] [rbp-110h]
  __int64 v19; // [rsp+0h] [rbp-110h]
  __int64 v20; // [rsp+8h] [rbp-108h]
  char *v21; // [rsp+8h] [rbp-108h]
  __m128i v22; // [rsp+10h] [rbp-100h] BYREF
  __m128i v23; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v24; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v25; // [rsp+40h] [rbp-D0h]
  __int64 v26; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  _OWORD v29[3]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v30; // [rsp+D0h] [rbp-40h]
  __int64 v31; // [rsp+D8h] [rbp-38h]

  v1 = (__int64 *)(a1 + 48);
  v3 = sub_1930F10(*(_QWORD **)(a1 + 24), *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 36));
  v20 = *(unsigned int *)(a1 + 12);
  v31 = sub_15AF870();
  v26 = 0;
  v4 = sub_19E7740(dest, &v26, dest[0].m128i_i8, (unsigned __int64)v29, v20);
  v5 = v26;
  v6 = v4;
  v7 = *(_QWORD *)(a1 + 40);
  v8 = v6 + 8;
  src = v7;
  if ( v6 + 8 <= (__int8 *)v29 )
  {
    *(_QWORD *)v6 = v7;
  }
  else
  {
    v18 = v26;
    memcpy(v6, &src, (char *)v29 - v6);
    if ( v18 )
    {
      sub_1593A20((unsigned __int64 *)v29, dest);
      v9 = (char *)((char *)v29 - v6);
      v10 = v18 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v22, dest[0].m128i_i64, v31);
      v16 = _mm_loadu_si128(&v23);
      v10 = 64;
      v17 = _mm_loadu_si128(&v24);
      v9 = (char *)((char *)v29 - v6);
      v29[0] = _mm_loadu_si128(&v22);
      v30 = v25;
      v29[1] = v16;
      v29[2] = v17;
    }
    v19 = v10;
    v21 = &dest[0].m128i_i8[8LL - (_QWORD)v9];
    if ( v21 > (char *)v29 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v9, 8LL - (_QWORD)v9);
    v8 = v21;
    v5 = v19;
  }
  src = v5;
  v11 = sub_19E7740(dest, &src, v8, (unsigned __int64)v29, v3);
  v12 = src;
  if ( src )
  {
    v15 = v11 - (__int8 *)dest;
    sub_19E16D0(dest[0].m128i_i8, v11, (char *)v29);
    sub_1593A20((unsigned __int64 *)v29, dest);
    v13 = sub_19E45B0(v29, v15 + v12);
  }
  else
  {
    v13 = sub_1593600(dest, v11 - (__int8 *)dest, v31);
  }
  dest[0].m128i_i64[0] = v13;
  return sub_19E8120(dest[0].m128i_i64, v1);
}
