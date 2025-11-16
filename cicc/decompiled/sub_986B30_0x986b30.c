// Function: sub_986B30
// Address: 0x986b30
//
__int64 __fastcall sub_986B30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v5; // eax
  __int64 v6; // rsi
  unsigned int v7; // ebx
  unsigned int v8; // r8d

  v5 = *((_DWORD *)a1 + 2);
  v6 = *a1;
  v7 = v5 - 1;
  if ( v5 <= 0x40 )
  {
    LOBYTE(a5) = v6 == 1LL << v7;
    return a5;
  }
  else
  {
    v8 = 0;
    if ( (*(_QWORD *)(v6 + 8LL * (v7 >> 6)) & (1LL << v7)) != 0 )
      LOBYTE(v8) = (unsigned int)sub_C44590(a1) == v7;
    return v8;
  }
}
