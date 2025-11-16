// Function: sub_38D7340
// Address: 0x38d7340
//
__int64 __fastcall sub_38D7340(__int64 a1, __int64 a2, _QWORD *a3, unsigned int *a4)
{
  __int64 v6; // r8
  _WORD *v7; // rdx
  _WORD *v8; // rsi
  __int16 v9; // cx
  unsigned int v10; // r15d
  __int64 (*v11)(); // rax
  int v12; // eax

  v6 = *(unsigned __int16 *)(*a3 + ((unsigned __int64)*a4 << 6) + 6);
  v7 = *(_WORD **)(a1 + 40);
  v8 = &v7[7 * v6];
  v9 = *v8 & 0x3FFF;
  if ( v9 == 0x3FFF )
    return 0;
  v10 = *(_DWORD *)(a1 + 28);
  if ( v9 == 16382 )
  {
    do
    {
      while ( 1 )
      {
        v11 = *(__int64 (**)())(*(_QWORD *)a2 + 16LL);
        if ( v11 != sub_168C390 )
          break;
        LODWORD(v6) = 0;
        v8 = v7;
        if ( (*v7 & 0x3FFF) != 0x3FFE )
          return sub_38D72B0(a2, (__int64)v8);
      }
      v12 = ((__int64 (__fastcall *)(__int64, _QWORD, unsigned int *, _QWORD))v11)(a2, (unsigned int)v6, a4, v10);
      v7 = *(_WORD **)(a1 + 40);
      LODWORD(v6) = v12;
      v8 = &v7[7 * v12];
    }
    while ( (*v8 & 0x3FFF) == 0x3FFE );
  }
  return sub_38D72B0(a2, (__int64)v8);
}
