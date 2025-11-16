// Function: sub_7D9DE0
// Address: 0x7d9de0
//
void __fastcall sub_7D9DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12

  v6 = *(_QWORD *)(a1 + 64);
  if ( *(_BYTE *)(a1 + 56) )
  {
    sub_8DD360(*(_QWORD *)(a1 + 64));
    if ( dword_4F077C4 == 2 )
      sub_7EAF80(v6);
  }
  else if ( dword_4F077C4 == 2 )
  {
    sub_7EE560(v6, 0);
  }
  else
  {
    sub_7D9DD0(*(_QWORD **)(a1 + 64), a2, a3, a4, a5, a6);
  }
}
