// Function: sub_169AA10
// Address: 0x169aa10
//
__int64 __fastcall sub_169AA10(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char v4; // cl
  int v5; // ebx
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v11; // rdx

  v3 = *(unsigned __int8 *)(a2 + 18);
  v4 = *(_BYTE *)(a2 + 18) & 7;
  if ( v4 == 1 )
  {
    v11 = *(_QWORD *)sub_16984A0(a2) & 0xFFFFFFFFFFFFFLL;
    v3 = *(unsigned __int8 *)(a2 + 18);
    v8 = v11;
    v9 = 0x7FF0000000000000LL;
  }
  else if ( v4 == 3 || !v4 )
  {
    v8 = 0;
    v9 = 0;
    if ( v4 != 3 )
      v9 = 0x7FF0000000000000LL;
  }
  else
  {
    v5 = *(__int16 *)(a2 + 16);
    v6 = *(_QWORD *)sub_16984A0(a2);
    v7 = v5 + 1023;
    if ( v7 == 1 )
    {
      v9 = v6 & 0x10000000000000LL;
      v8 = v6 & 0xFFFFFFFFFFFFFLL;
    }
    else
    {
      v8 = v6 & 0xFFFFFFFFFFFFFLL;
      v9 = (v7 << 52) & 0x7FF0000000000000LL;
    }
    v3 = *(unsigned __int8 *)(a2 + 18);
  }
  *(_DWORD *)(a1 + 8) = 64;
  LOBYTE(v3) = (unsigned __int8)v3 >> 3;
  *(_QWORD *)a1 = v8 | v9 | (v3 << 63);
  return a1;
}
