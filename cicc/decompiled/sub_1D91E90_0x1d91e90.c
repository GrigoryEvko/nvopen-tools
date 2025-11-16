// Function: sub_1D91E90
// Address: 0x1d91e90
//
__int64 __fastcall sub_1D91E90(__int64 a1, char *a2, _DWORD *a3)
{
  char v3; // al
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (*v9)(); // rax

  *a3 = 0;
  v3 = *a2;
  if ( (*a2 & 2) != 0 || (v3 & 1) != 0 || (v3 & 0x10) != 0 )
    return 0;
  result = 1;
  v7 = *((_QWORD *)a2 + 2);
  if ( (unsigned int)((__int64)(*(_QWORD *)(v7 + 72) - *(_QWORD *)(v7 + 64)) >> 3) > 1 )
  {
    if ( (a2[1] & 1) != 0 )
      return 0;
    v8 = *(_QWORD *)(a1 + 544);
    v9 = *(__int64 (**)())(*(_QWORD *)v8 + 344LL);
    if ( v9 == sub_1D91890 )
      return 0;
    result = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v9)(v8, v7, *((unsigned int *)a2 + 1));
    if ( !(_BYTE)result )
      return 0;
    else
      *a3 = *((_DWORD *)a2 + 1);
  }
  return result;
}
