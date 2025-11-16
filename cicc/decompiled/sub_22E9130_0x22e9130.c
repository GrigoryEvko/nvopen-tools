// Function: sub_22E9130
// Address: 0x22e9130
//
void __fastcall sub_22E9130(__int64 a1, __int64 *a2, char a3, void **a4)
{
  __int64 v4; // r12
  __int64 v5; // r13
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rdi
  _WORD *v9; // rdx
  __int64 v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 *v11; // [rsp+8h] [rbp-48h]
  char v12; // [rsp+10h] [rbp-40h]
  char v13; // [rsp+11h] [rbp-3Fh]
  _QWORD *v14[2]; // [rsp+20h] [rbp-30h] BYREF
  __int64 v15; // [rsp+30h] [rbp-20h] BYREF

  v10 = a1;
  v13 = a3;
  v11 = a2;
  v12 = 0;
  sub_CA0F50((__int64 *)v14, a4);
  sub_22E6E00(&v10, v14);
  sub_22E8D90((__int64)&v10);
  v4 = v10;
  v5 = *v11;
  v6 = *(__m128i **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v6 <= 0x19u )
  {
    sub_CB6200(v10, "\tcolorscheme = \"paired12\"\n", 0x1Au);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428CE20);
    qmemcpy(&v6[1], "paired12\"\n", 10);
    *v6 = si128;
    *(_QWORD *)(v4 + 32) += 26LL;
  }
  sub_22E61E0(*(__int64 **)(v5 + 32), v4, 4);
  v8 = v10;
  v9 = *(_WORD **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v9 <= 1u )
  {
    sub_CB6200(v10, "}\n", 2u);
  }
  else
  {
    *v9 = 2685;
    *(_QWORD *)(v8 + 32) += 2LL;
  }
  if ( v14[0] != &v15 )
    j_j___libc_free_0((unsigned __int64)v14[0]);
}
