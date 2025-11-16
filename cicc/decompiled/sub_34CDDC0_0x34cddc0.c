// Function: sub_34CDDC0
// Address: 0x34cddc0
//
__int64 __fastcall sub_34CDDC0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 v4; // [rsp+8h] [rbp-18h]

  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 24) + 56LL))(*(_QWORD *)(a1 + 24));
  if ( BYTE4(v2) )
  {
    v4 = v2;
  }
  else if ( a2 > 1 )
  {
    BUG();
  }
  BYTE4(v4) = BYTE4(v2);
  return v4;
}
