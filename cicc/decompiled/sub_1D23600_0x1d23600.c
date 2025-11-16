// Function: sub_1D23600
// Address: 0x1d23600
//
__int64 __fastcall sub_1D23600(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int16 v3; // ax

  v2 = *(unsigned __int16 *)(a2 + 24);
  if ( v2 == 10 || v2 == 32 || (unsigned __int8)sub_1D168E0(a2) )
    return a2;
  if ( ((v3 = *(_WORD *)(a2 + 24), (unsigned __int16)(v3 - 34) <= 1u) || (unsigned __int16)(v3 - 12) <= 1u)
    && v3 == 12
    && (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 1048LL))(*(_QWORD *)(a1 + 16), a2) )
  {
    return a2;
  }
  else
  {
    return 0;
  }
}
