// Function: sub_3249DD0
// Address: 0x3249dd0
//
void __fastcall sub_3249DD0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al

  v3 = *(_BYTE *)(a3 - 16);
  if ( (v3 & 2) != 0 )
    sub_3249CA0(a1, a2, *(_DWORD *)(a3 + 4), *(_QWORD *)(*(_QWORD *)(a3 - 32) + 16LL));
  else
    sub_3249CA0(a1, a2, *(_DWORD *)(a3 + 4), *(_QWORD *)(a3 - 8LL * ((v3 >> 2) & 0xF)));
}
