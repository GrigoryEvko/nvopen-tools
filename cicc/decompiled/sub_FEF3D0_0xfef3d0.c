// Function: sub_FEF3D0
// Address: 0xfef3d0
//
bool __fastcall sub_FEF3D0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4[2]; // [rsp+0h] [rbp-10h] BYREF

  v2 = *a2;
  v4[0] = a2[1];
  v4[1] = v2;
  return sub_FEF380(a1, v4);
}
