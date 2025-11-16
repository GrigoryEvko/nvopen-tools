// Function: sub_3249E50
// Address: 0x3249e50
//
void __fastcall sub_3249E50(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al

  v3 = *(_BYTE *)(a3 - 16);
  if ( (v3 & 2) != 0 )
    sub_3249CA0(a1, a2, *(_DWORD *)(a3 + 16), *(_QWORD *)(*(_QWORD *)(a3 - 32) + 8LL));
  else
    sub_3249CA0(a1, a2, *(_DWORD *)(a3 + 16), *(_QWORD *)(a3 - 16 - 8LL * ((v3 >> 2) & 0xF) + 8));
}
