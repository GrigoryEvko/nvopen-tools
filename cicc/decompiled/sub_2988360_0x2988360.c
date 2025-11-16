// Function: sub_2988360
// Address: 0x2988360
//
__int64 __fastcall sub_2988360(__int64 *a1, _QWORD *a2)
{
  __int64 *v3; // r12
  __int64 v4; // rax
  __int64 **v5; // rdi
  __int64 result; // rax

  v3 = (__int64 *)sub_AA48A0(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  *a1 = sub_BCB2A0(v3);
  a1[1] = sub_ACD6D0(v3);
  v4 = sub_ACD720(v3);
  v5 = (__int64 **)*a1;
  a1[2] = v4;
  result = sub_ACADE0(v5);
  a1[6] = 0;
  a1[3] = result;
  return result;
}
