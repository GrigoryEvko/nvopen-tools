// Function: sub_19E8120
// Address: 0x19e8120
//
unsigned __int64 __fastcall sub_19E8120(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int8 *v4; // rax
  __int64 v5; // r14
  __int8 *v6; // rdi
  __int64 v7; // rax
  char *v8; // r15
  char *v9; // rbx
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // [rsp+10h] [rbp-100h] BYREF
  __m128i v14; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v15; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v16; // [rsp+40h] [rbp-D0h]
  __int64 v17; // [rsp+50h] [rbp-C0h] BYREF
  __int64 src; // [rsp+58h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+60h] [rbp-B0h] BYREF
  _OWORD v20[3]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v21; // [rsp+D0h] [rbp-40h]
  unsigned __int64 v22; // [rsp+D8h] [rbp-38h]

  v2 = sub_15AF870();
  v3 = *a1;
  v22 = v2;
  v17 = 0;
  v4 = sub_19E7740(dest, &v17, dest[0].m128i_i8, (unsigned __int64)v20, v3);
  v5 = v17;
  v6 = v4;
  v7 = *a2;
  v8 = v6 + 8;
  src = *a2;
  if ( v6 + 8 <= (__int8 *)v20 )
  {
    *(_QWORD *)v6 = v7;
  }
  else
  {
    v9 = (char *)((char *)v20 - v6);
    memcpy(v6, &src, (char *)v20 - v6);
    if ( v5 )
    {
      v5 += 64;
      sub_1593A20((unsigned __int64 *)v20, dest);
    }
    else
    {
      v5 = 64;
      sub_15938B0((unsigned __int64 *)&v13, dest[0].m128i_i64, v22);
      v11 = _mm_loadu_si128(&v14);
      v12 = _mm_loadu_si128(&v15);
      v20[0] = _mm_loadu_si128(&v13);
      v21 = v16;
      v20[1] = v11;
      v20[2] = v12;
    }
    v8 = &dest[0].m128i_i8[8LL - (_QWORD)v9];
    if ( v8 > (char *)v20 )
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v9, 8LL - (_QWORD)v9);
  }
  if ( !v5 )
    return sub_1593600(dest, v8 - (char *)dest, v22);
  sub_19E16D0(dest[0].m128i_i8, v8, (char *)v20);
  sub_1593A20((unsigned __int64 *)v20, dest);
  return sub_19E45B0(v20, v5 + v8 - (char *)dest);
}
