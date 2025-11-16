// Function: sub_2FEC400
// Address: 0x2fec400
//
__int64 __fastcall sub_2FEC400(__int64 a1, unsigned int a2, __int64 a3, __int64 *a4)
{
  char *v5; // rax
  size_t v6; // rdx
  __int64 v7; // r9
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v9[0] = sub_B2D7E0(*a4, "reciprocal-estimates", 0x14u);
  v5 = (char *)sub_A72240(v9);
  return sub_2FE4940(1u, a2, a3, v5, v6, v7);
}
