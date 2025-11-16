// Function: sub_33E3830
// Address: 0x33e3830
//
bool __fastcall sub_33E3830(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  if ( !*(_DWORD *)(a1 + 16) && *(_QWORD *)(a2 + 80) )
    return 0;
  sub_33E3600(a2, a5, a3, a4, a5, a1);
  return sub_C656C0(a5, a3);
}
