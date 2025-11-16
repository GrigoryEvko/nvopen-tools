// Function: sub_1F01F20
// Address: 0x1f01f20
//
void __fastcall sub_1F01F20(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  if ( (*(_BYTE *)(a1 + 236) & 1) == 0 )
    sub_1F01DD0(a1, a2, a3, a4, a5, a6);
  if ( (unsigned int)a2 > *(_DWORD *)(a1 + 240) )
  {
    sub_1F01800(a1, (__int64)a2, a3, a4, a5, a6);
    *(_BYTE *)(a1 + 236) |= 1u;
    *(_DWORD *)(a1 + 240) = (_DWORD)a2;
  }
}
