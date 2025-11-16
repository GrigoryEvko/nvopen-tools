// Function: sub_B326A0
// Address: 0xb326a0
//
__int64 __fastcall sub_B326A0(__int64 a1)
{
  __int64 v1; // rax

  if ( *(_BYTE *)a1 == 1 )
  {
    v1 = sub_B325F0(a1);
    if ( v1 )
      return *(_QWORD *)(v1 + 48);
    else
      return 0;
  }
  else if ( *(_BYTE *)a1 == 2 )
  {
    return 0;
  }
  else
  {
    return *(_QWORD *)(a1 + 48);
  }
}
