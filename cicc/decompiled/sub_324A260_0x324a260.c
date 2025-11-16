// Function: sub_324A260
// Address: 0x324a260
//
__int64 __fastcall sub_324A260(__int64 *a1, __int64 a2, __int16 a3, __int64 a4, char a5)
{
  unsigned int v5; // eax
  unsigned __int64 **v6; // rsi
  __int64 v8; // r8
  __int64 v9; // [rsp-8h] [rbp-8h]

  v5 = *(_DWORD *)(a4 + 8);
  if ( v5 > 0x40 )
    return sub_324A160(a1, a2, a3, (__int64 *)a4);
  v6 = (unsigned __int64 **)(a2 + 8);
  if ( a5 )
  {
    *((_BYTE *)&v9 - 2) = 0;
    return sub_3249A20(a1, v6, a3, *((_DWORD *)&v9 - 1), *(_QWORD *)a4);
  }
  else
  {
    v8 = 0;
    if ( v5 )
      v8 = (__int64)(*(_QWORD *)a4 << (64 - (unsigned __int8)v5)) >> (64 - (unsigned __int8)v5);
    *((_BYTE *)&v9 - 2) = 0;
    return sub_32498F0(a1, v6, a3, *((_DWORD *)&v9 - 1), v8);
  }
}
