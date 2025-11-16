// Function: sub_2F8F720
// Address: 0x2f8f720
//
void __fastcall sub_2F8F720(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  if ( (*(_BYTE *)(a1 + 254) & 1) == 0 )
    sub_2F8F5D0(a1, a2, a3, a4, a5, a6);
  if ( (unsigned int)a2 > *(_DWORD *)(a1 + 240) )
  {
    sub_2F8EFB0(a1, (__int64)a2, a3, a4, a5, a6);
    *(_BYTE *)(a1 + 254) |= 1u;
    *(_DWORD *)(a1 + 240) = (_DWORD)a2;
  }
}
