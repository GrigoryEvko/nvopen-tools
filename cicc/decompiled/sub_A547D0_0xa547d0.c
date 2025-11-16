// Function: sub_A547D0
// Address: 0xa547d0
//
__int64 __fastcall sub_A547D0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdi

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
  {
    a2 = (unsigned int)a2;
    v3 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL * (unsigned int)a2);
    if ( v3 )
      return sub_B91420(v3, a2);
  }
  else
  {
    a2 = (unsigned int)a2;
    v3 = *(_QWORD *)(a1 - 16 - 8LL * ((v2 >> 2) & 0xF) + 8LL * (unsigned int)a2);
    if ( v3 )
      return sub_B91420(v3, a2);
  }
  return 0;
}
