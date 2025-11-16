// Function: sub_164ECF0
// Address: 0x164ecf0
//
__int64 __fastcall sub_164ECF0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax

  v2 = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)v2 >= *(_QWORD *)(a1 + 16) )
  {
    a1 = sub_16E7DE0(a1, 32);
  }
  else
  {
    *(_QWORD *)(a1 + 24) = v2 + 1;
    *v2 = 32;
  }
  return sub_154E060(a2, a1, 0, 0);
}
