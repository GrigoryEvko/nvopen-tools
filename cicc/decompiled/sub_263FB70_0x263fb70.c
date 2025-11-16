// Function: sub_263FB70
// Address: 0x263fb70
//
__int64 __fastcall sub_263FB70(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 *v3; // r12
  __int64 result; // rax

  a1[1] = 8;
  v1 = sub_22077B0(0x40u);
  v2 = a1[1];
  *a1 = v1;
  v3 = (__int64 *)(v1 + ((4 * v2 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  result = sub_22077B0(0x200u);
  a1[5] = (__int64)v3;
  *v3 = result;
  a1[9] = (__int64)v3;
  a1[3] = result;
  a1[4] = result + 512;
  a1[7] = result;
  a1[8] = result + 512;
  a1[2] = result;
  a1[6] = result;
  return result;
}
