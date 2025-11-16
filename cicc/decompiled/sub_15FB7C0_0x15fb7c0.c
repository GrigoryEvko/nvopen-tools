// Function: sub_15FB7C0
// Address: 0x15fb7c0
//
__int64 __fastcall sub_15FB7C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx

  v4 = *(_QWORD *)(a1 - 48);
  if ( *(_BYTE *)(v4 + 16) <= 0x10u )
  {
    v5 = *(_QWORD *)(a1 - 24);
    if ( sub_1596070(*(_QWORD *)(a1 - 48), a2, a3, a4) )
      return v5;
  }
  return v4;
}
