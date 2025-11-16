// Function: sub_DFABC0
// Address: 0xdfabc0
//
__int64 __fastcall sub_DFABC0(__int64 a1, __int64 *a2, unsigned __int8 a3, unsigned __int8 a4)
{
  __int64 v4; // rsi
  __int64 result; // rax
  __int64 (__fastcall *v6)(__int64); // r8

  v4 = *a2;
  result = a1;
  v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 840LL);
  if ( v6 == sub_DF5F00 )
  {
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = a1 + 24;
    *(_QWORD *)(a1 + 16) = 0x800000000LL;
    *(_DWORD *)(a1 + 56) = 1;
    *(_BYTE *)(a1 + 60) = 0;
    *(_QWORD *)(a1 + 64) = a1 + 80;
    *(_QWORD *)(a1 + 72) = 0x400000000LL;
  }
  else
  {
    ((void (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v6)(a1, v4, a3, a4);
    return a1;
  }
  return result;
}
