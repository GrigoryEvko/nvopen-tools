// Function: sub_14758B0
// Address: 0x14758b0
//
__int64 __fastcall sub_14758B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx

  v4 = sub_1456040(a2);
  v5 = sub_1456C90(a1, v4);
  if ( v5 == sub_1456C90(a1, a3) )
    return a2;
  else
    return sub_14747F0(a1, a2, a3, 0);
}
