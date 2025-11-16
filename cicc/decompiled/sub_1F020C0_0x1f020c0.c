// Function: sub_1F020C0
// Address: 0x1f020c0
//
void __fastcall sub_1F020C0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  if ( (*(_BYTE *)(a1 + 236) & 2) == 0 )
    sub_1F01F70(a1, a2, a3, a4, a5, a6);
  if ( (unsigned int)a2 > *(_DWORD *)(a1 + 244) )
  {
    sub_1F01900(a1, (__int64)a2, a3, a4, a5, a6);
    *(_BYTE *)(a1 + 236) |= 2u;
    *(_DWORD *)(a1 + 244) = (_DWORD)a2;
  }
}
