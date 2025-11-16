// Function: sub_19E82B0
// Address: 0x19e82b0
//
unsigned __int64 __fastcall sub_19E82B0(__int64 a1)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r14
  __int8 *v5; // rax
  __int64 v6; // rcx
  __int8 *v7; // rdi
  __int64 v8; // rax
  char *v9; // r15
  char *v10; // r8
  __int64 v11; // rcx
  __int8 *v12; // rax
  __int64 v13; // r14
  signed __int64 v15; // rbx
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // [rsp+0h] [rbp-110h]
  __int64 v19; // [rsp+8h] [rbp-108h]
  __m128i v20; // [rsp+10h] [rbp-100h] BYREF
  __m128i v21; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v22; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v23; // [rsp+40h] [rbp-D0h]
  __int64 v24; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  _OWORD v27[3]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v28; // [rsp+D0h] [rbp-40h]
  __int64 v29; // [rsp+D8h] [rbp-38h]

  v2 = sub_1930F10(*(_QWORD **)(a1 + 24), *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 36));
  v3 = *(unsigned int *)(a1 + 12);
  v4 = v2;
  v29 = sub_15AF870();
  v24 = 0;
  v5 = sub_19E7740(dest, &v24, dest[0].m128i_i8, (unsigned __int64)v27, v3);
  v6 = v24;
  v7 = v5;
  v8 = *(_QWORD *)(a1 + 40);
  v9 = v7 + 8;
  src = v8;
  if ( v7 + 8 <= (__int8 *)v27 )
  {
    *(_QWORD *)v7 = v8;
  }
  else
  {
    v18 = v24;
    memcpy(v7, &src, (char *)v27 - v7);
    if ( v18 )
    {
      sub_1593A20((unsigned __int64 *)v27, dest);
      v10 = (char *)((char *)v27 - v7);
      v11 = v18 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v20, dest[0].m128i_i64, v29);
      v16 = _mm_loadu_si128(&v21);
      v11 = 64;
      v17 = _mm_loadu_si128(&v22);
      v10 = (char *)((char *)v27 - v7);
      v27[0] = _mm_loadu_si128(&v20);
      v28 = v23;
      v27[1] = v16;
      v27[2] = v17;
    }
    v19 = v11;
    v9 = &dest[0].m128i_i8[8LL - (_QWORD)v10];
    if ( v9 > (char *)v27 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v10, 8LL - (_QWORD)v10);
    v6 = v19;
  }
  src = v6;
  v12 = sub_19E7740(dest, &src, v9, (unsigned __int64)v27, v4);
  v13 = src;
  if ( !src )
    return sub_1593600(dest, v12 - (__int8 *)dest, v29);
  v15 = v12 - (__int8 *)dest;
  sub_19E16D0(dest[0].m128i_i8, v12, (char *)v27);
  sub_1593A20((unsigned __int64 *)v27, dest);
  return sub_19E45B0(v27, v15 + v13);
}
