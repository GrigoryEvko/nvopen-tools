// Function: sub_15FB6B0
// Address: 0x15fb6b0
//
char __fastcall sub_15FB6B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdi

  if ( *(_BYTE *)(a1 + 16) == 37 && (v4 = *(_QWORD *)(a1 - 48), *(_BYTE *)(v4 + 16) <= 0x10u) )
    return sub_1595D90(v4, a2, a3, a4);
  else
    return 0;
}
