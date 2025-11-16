// Function: sub_1683C60
// Address: 0x1683c60
//
__int64 __fastcall sub_1683C60(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8

  v1 = sub_1689050();
  v2 = *(_QWORD *)(v1 + 24);
  *(_QWORD *)(v1 + 24) = a1;
  return v2;
}
