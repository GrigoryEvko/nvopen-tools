// Function: sub_31A4BE0
// Address: 0x31a4be0
//
__int64 __fastcall sub_31A4BE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  int v8; // eax
  unsigned int v9; // r13d

  v6 = (unsigned __int8)byte_5035428;
  if ( !byte_5035428 )
    return v6;
  v8 = *(_DWORD *)(a1 + 40);
  v9 = *(_DWORD *)(a1 + 8);
  if ( v8 == -1 )
  {
    if ( (unsigned __int8)sub_F6E590(*(_QWORD *)(a1 + 104), a2, a3, a4, a5, a6) )
      goto LABEL_5;
    v8 = *(_DWORD *)(a1 + 40);
  }
  if ( v8 == 1 )
    return v6;
LABEL_5:
  LOBYTE(v6) = v9 > 1;
  return v6;
}
