// Function: sub_3143F30
// Address: 0x3143f30
//
__int64 __fastcall sub_3143F30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( *(_QWORD *)(a2 + 48) )
  {
    v2 = sub_B10CD0(a2 + 48);
    sub_3143E70(a1, v2);
  }
  else
  {
    *(_BYTE *)(a1 + 20) = 0;
  }
  return a1;
}
