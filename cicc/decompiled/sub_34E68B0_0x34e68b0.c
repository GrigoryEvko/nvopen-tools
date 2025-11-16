// Function: sub_34E68B0
// Address: 0x34e68b0
//
__int64 __fastcall sub_34E68B0(__int64 a1, char *a2, _DWORD *a3)
{
  char v3; // al
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 (*v8)(); // rax

  *a3 = 0;
  v3 = *a2;
  if ( (*a2 & 2) != 0 || (v3 & 1) != 0 || (v3 & 0x10) != 0 )
    return 0;
  result = 1;
  v6 = *((_QWORD *)a2 + 2);
  if ( *(_DWORD *)(v6 + 72) > 1u )
  {
    if ( (a2[1] & 1) != 0 )
      return 0;
    v7 = *(_QWORD *)(a1 + 528);
    v8 = *(__int64 (**)())(*(_QWORD *)v7 + 432LL);
    if ( v8 == sub_2FDC550 )
      return 0;
    result = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v8)(v7, v6, *((unsigned int *)a2 + 1));
    if ( !(_BYTE)result )
      return 0;
    else
      *a3 = *((_DWORD *)a2 + 1);
  }
  return result;
}
