// Function: sub_1F3BEE0
// Address: 0x1f3bee0
//
__int64 __fastcall sub_1F3BEE0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, _DWORD *a5)
{
  __int64 (*v5)(void); // r9
  __int64 result; // rax
  __int64 v7; // rdx

  if ( **(_WORD **)(a2 + 16) == 8 )
  {
    v7 = *(_QWORD *)(a2 + 32);
    result = 0;
    if ( (*(_BYTE *)(v7 + 84) & 1) == 0 )
    {
      *a4 = *(_DWORD *)(v7 + 48);
      a4[1] = (*(_DWORD *)(v7 + 40) >> 8) & 0xFFF;
      *a5 = *(_DWORD *)(v7 + 88);
      a5[1] = (*(_DWORD *)(v7 + 80) >> 8) & 0xFFF;
      a5[2] = *(_QWORD *)(v7 + 144);
      return 1;
    }
  }
  else
  {
    v5 = *(__int64 (**)(void))(*(_QWORD *)a1 + 536LL);
    result = 0;
    if ( v5 != sub_1F394C0 )
      return v5();
  }
  return result;
}
