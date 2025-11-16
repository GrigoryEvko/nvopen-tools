// Function: sub_2F8F8C0
// Address: 0x2f8f8c0
//
void __fastcall sub_2F8F8C0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  if ( (*(_BYTE *)(a1 + 254) & 2) == 0 )
    sub_2F8F770(a1, a2, a3, a4, a5, a6);
  if ( (unsigned int)a2 > *(_DWORD *)(a1 + 244) )
  {
    sub_2F8F0B0(a1, (__int64)a2, a3, a4, a5, a6);
    *(_BYTE *)(a1 + 254) |= 2u;
    *(_DWORD *)(a1 + 244) = (_DWORD)a2;
  }
}
