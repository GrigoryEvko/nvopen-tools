// Function: sub_2506070
// Address: 0x2506070
//
__int64 __fastcall sub_2506070(__int64 a1, __int64 a2)
{
  int v2; // edx

  ++**(_DWORD **)a1;
  v2 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v2 - 17) <= 1 )
    LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( (_BYTE)v2 == 14 )
    return sub_B2D640(**(_QWORD **)(a1 + 8), **(_DWORD **)a1, 50);
  else
    return 1;
}
