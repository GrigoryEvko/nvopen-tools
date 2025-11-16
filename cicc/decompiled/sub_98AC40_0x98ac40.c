// Function: sub_98AC40
// Address: 0x98ac40
//
__int64 __fastcall sub_98AC40(__int64 a1, char a2)
{
  __int64 v2; // r12

  v2 = sub_B494D0(a1, 52);
  if ( v2 || !sub_98AB90(a1, a2) )
    return v2;
  else
    return *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
}
