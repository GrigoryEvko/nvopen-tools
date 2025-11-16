// Function: sub_B50700
// Address: 0xb50700
//
__int64 __fastcall sub_B50700(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v5; // eax
  unsigned int v6; // r8d

  v5 = *a1 - 29;
  if ( v5 <= 0x28 )
  {
    LOBYTE(a5) = v5 > 0x25;
    return a5;
  }
  else
  {
    v6 = 0;
    if ( *a1 == 78 && *(_BYTE *)(*(_QWORD *)(*((_QWORD *)a1 - 4) + 8LL) + 8LL) == 12 )
      LOBYTE(v6) = *(_BYTE *)(*((_QWORD *)a1 + 1) + 8LL) == 12;
    return v6;
  }
}
