// Function: sub_2E86580
// Address: 0x2e86580
//
__int64 __fastcall sub_2E86580(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d
  unsigned __int8 *v3; // rdx

  v1 = *(_QWORD *)(a1 + 48);
  v2 = 0;
  v3 = (unsigned __int8 *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v1 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v1 & 7) == 3 && v3[8] )
    return *(unsigned int *)&v3[8 * *(int *)v3 + 16 + 8 * v3[7] + 8 * v3[6] + 8 * (__int64)(v3[5] + v3[4])];
  return v2;
}
