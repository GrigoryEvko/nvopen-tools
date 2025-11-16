// Function: sub_19E8970
// Address: 0x19e8970
//
unsigned __int64 __fastcall sub_19E8970(__int64 a1)
{
  unsigned __int64 v2; // r13
  __int8 *v3; // rax
  __int64 v4; // rcx
  __int8 *v5; // rdi
  __int64 v6; // rax
  char *v7; // r10
  char *v8; // r8
  __int64 v9; // rcx
  __int8 *v10; // r13
  unsigned __int64 v11; // r13
  __int8 *v12; // rax
  __int8 *v13; // rax
  __int64 v14; // r13
  signed __int64 v16; // rbx
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __int64 v19; // [rsp+8h] [rbp-118h]
  __int64 v20; // [rsp+8h] [rbp-118h]
  __int64 v21; // [rsp+10h] [rbp-110h]
  char *v22; // [rsp+10h] [rbp-110h]
  __int64 v23; // [rsp+10h] [rbp-110h]
  unsigned __int64 v24; // [rsp+18h] [rbp-108h]
  __m128i v25; // [rsp+20h] [rbp-100h] BYREF
  __m128i v26; // [rsp+30h] [rbp-F0h] BYREF
  __m128i v27; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+50h] [rbp-D0h]
  __int64 v29; // [rsp+60h] [rbp-C0h] BYREF
  __int64 src; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  _OWORD v32[3]; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v33; // [rsp+E0h] [rbp-40h]
  __int64 v34; // [rsp+E8h] [rbp-38h]

  v24 = sub_1597510(*(__int64 **)(a1 + 56), *(_QWORD *)(a1 + 56) + 4LL * *(unsigned int *)(a1 + 52));
  v2 = sub_1930F10(*(_QWORD **)(a1 + 24), *(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 36));
  v21 = *(unsigned int *)(a1 + 12);
  v34 = sub_15AF870();
  v29 = 0;
  v3 = sub_19E7740(dest, &v29, dest[0].m128i_i8, (unsigned __int64)v32, v21);
  v4 = v29;
  v5 = v3;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = v5 + 8;
  src = v6;
  if ( v5 + 8 <= (__int8 *)v32 )
  {
    *(_QWORD *)v5 = v6;
  }
  else
  {
    v19 = v29;
    memcpy(v5, &src, (char *)v32 - v5);
    if ( v19 )
    {
      sub_1593A20((unsigned __int64 *)v32, dest);
      v8 = (char *)((char *)v32 - v5);
      v9 = v19 + 64;
    }
    else
    {
      sub_15938B0((unsigned __int64 *)&v25, dest[0].m128i_i64, v34);
      v17 = _mm_loadu_si128(&v26);
      v9 = 64;
      v18 = _mm_loadu_si128(&v27);
      v8 = (char *)((char *)v32 - v5);
      v32[0] = _mm_loadu_si128(&v25);
      v33 = v28;
      v32[1] = v17;
      v32[2] = v18;
    }
    v20 = v9;
    v22 = &dest[0].m128i_i8[8LL - (_QWORD)v8];
    if ( v22 > (char *)v32 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v8, 8LL - (_QWORD)v8);
    v7 = v22;
    v4 = v20;
  }
  src = v4;
  v10 = sub_19E7740(dest, &src, v7, (unsigned __int64)v32, v2);
  if ( src )
  {
    v23 = src;
    sub_19E16D0(dest[0].m128i_i8, v10, (char *)v32);
    sub_1593A20((unsigned __int64 *)v32, dest);
    v11 = sub_19E45B0(v32, v10 - (__int8 *)dest + v23);
  }
  else
  {
    v11 = sub_1593600(dest, v10 - (__int8 *)dest, v34);
  }
  v34 = sub_15AF870();
  v29 = 0;
  v12 = sub_19E7740(dest, &v29, dest[0].m128i_i8, (unsigned __int64)v32, v11);
  src = v29;
  v13 = sub_19E7740(dest, &src, v12, (unsigned __int64)v32, v24);
  v14 = src;
  if ( !src )
    return sub_1593600(dest, v13 - (__int8 *)dest, v34);
  v16 = v13 - (__int8 *)dest;
  sub_19E16D0(dest[0].m128i_i8, v13, (char *)v32);
  sub_1593A20((unsigned __int64 *)v32, dest);
  return sub_19E45B0(v32, v16 + v14);
}
