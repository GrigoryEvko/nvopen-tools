// Function: sub_986BE0
// Address: 0x986be0
//
__int64 __fastcall sub_986BE0(__int64 a1, __int64 a2)
{
  if ( *(_DWORD *)(a1 + 8) <= 0x40u && *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    return a1;
  }
  else
  {
    sub_C43990(a1, a2);
    return a1;
  }
}
