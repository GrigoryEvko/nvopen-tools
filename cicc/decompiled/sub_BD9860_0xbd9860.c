// Function: sub_BD9860
// Address: 0xbd9860
//
__int64 __fastcall sub_BD9860(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax

  v2 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v2 >= *(_QWORD *)(a1 + 24) )
  {
    a1 = sub_CB5D20(a1, 32);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v2 + 1;
    *v2 = 32;
  }
  return sub_A587F0(a2, a1, 0, 0);
}
