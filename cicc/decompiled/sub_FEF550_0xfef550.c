// Function: sub_FEF550
// Address: 0xfef550
//
__int64 __fastcall sub_FEF550(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8

  v3 = *(_QWORD *)(a2 + 8);
  if ( v3 )
    return sub_D472F0(v3, a3);
  else
    return sub_FEEFB0(*(_QWORD *)(a1 + 80), *(_DWORD *)(a2 + 16), a3);
}
