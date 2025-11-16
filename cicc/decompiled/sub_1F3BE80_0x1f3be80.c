// Function: sub_1F3BE80
// Address: 0x1f3be80
//
__int64 __fastcall sub_1F3BE80(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 (*v4)(void); // r8
  __int64 result; // rax
  __int64 v6; // rdx

  if ( **(_WORD **)(a2 + 16) == 7 )
  {
    v6 = *(_QWORD *)(a2 + 32);
    result = 0;
    if ( (*(_BYTE *)(v6 + 44) & 1) == 0 )
    {
      *a4 = *(_DWORD *)(v6 + 48);
      a4[1] = (*(_DWORD *)(v6 + 40) >> 8) & 0xFFF;
      a4[2] = *(_QWORD *)(v6 + 104);
      return 1;
    }
  }
  else
  {
    v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 528LL);
    result = 0;
    if ( v4 != sub_1F394B0 )
      return v4();
  }
  return result;
}
