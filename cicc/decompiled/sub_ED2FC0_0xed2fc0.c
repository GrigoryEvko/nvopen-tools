// Function: sub_ED2FC0
// Address: 0xed2fc0
//
__int64 __fastcall sub_ED2FC0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8
  unsigned __int64 v3; // rax

  v1 = sub_ED2E40(a1);
  v2 = 40;
  v3 = v1 - 8;
  if ( v3 <= 4 )
    return *(_QWORD *)&a0_1[8 * v3];
  return v2;
}
