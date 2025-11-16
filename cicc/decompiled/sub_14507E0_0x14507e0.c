// Function: sub_14507E0
// Address: 0x14507e0
//
__int64 __fastcall sub_14507E0(__int64 a1, __int64 *a2, char a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v6; // r14
  __m128i *v7; // rdx
  __m128i si128; // xmm0
  __int64 v9; // rdi
  _WORD *v10; // rdx
  __int64 v12; // [rsp+0h] [rbp-60h] BYREF
  __int64 *v13; // [rsp+8h] [rbp-58h]
  char v14; // [rsp+10h] [rbp-50h]
  __int64 v15[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v16; // [rsp+30h] [rbp-30h] BYREF

  v12 = a1;
  v14 = a3;
  v13 = a2;
  sub_16E2FC0(v15, a4);
  sub_144E9D0(&v12, v15);
  sub_1450450(&v12);
  v5 = v12;
  v6 = *v13;
  v7 = *(__m128i **)(v12 + 24);
  if ( *(_QWORD *)(v12 + 16) - (_QWORD)v7 <= 0x19u )
  {
    sub_16E7EE0(v12, "\tcolorscheme = \"paired12\"\n", 26);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428CE20);
    qmemcpy(&v7[1], "paired12\"\n", 10);
    *v7 = si128;
    *(_QWORD *)(v5 + 24) += 26LL;
  }
  sub_144DDD0(*(__int64 **)(v6 + 32), v5, 4);
  v9 = v12;
  v10 = *(_WORD **)(v12 + 24);
  if ( *(_QWORD *)(v12 + 16) - (_QWORD)v10 <= 1u )
  {
    sub_16E7EE0(v12, "}\n", 2);
  }
  else
  {
    *v10 = 2685;
    *(_QWORD *)(v9 + 24) += 2LL;
  }
  if ( (__int64 *)v15[0] != &v16 )
    j_j___libc_free_0(v15[0], v16 + 1);
  return a1;
}
