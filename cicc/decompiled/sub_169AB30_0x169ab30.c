// Function: sub_169AB30
// Address: 0x169ab30
//
__int64 __fastcall sub_169AB30(__int64 a1, __int64 a2)
{
  int v3; // eax
  char v4; // cl
  int v5; // ebx
  __int64 v6; // rsi
  int v7; // esi
  int v8; // edx
  int v10; // edx
  __int64 v11; // rdx
  int v12; // esi

  v3 = *(unsigned __int8 *)(a2 + 18);
  v4 = *(_BYTE *)(a2 + 18) & 7;
  if ( v4 == 1 )
  {
    v11 = *(_QWORD *)sub_16984A0(a2);
    v3 = *(unsigned __int8 *)(a2 + 18);
    v12 = v11;
    v8 = 2139095040;
    v7 = v12 & 0x7FFFFF;
  }
  else if ( v4 == 3 || !v4 )
  {
    v7 = 0;
    v8 = 0;
    if ( v4 != 3 )
      v8 = 2139095040;
  }
  else
  {
    v5 = *(__int16 *)(a2 + 16) + 127;
    v6 = *(_QWORD *)sub_16984A0(a2);
    if ( v5 == 1 )
    {
      v10 = v6;
      v3 = *(unsigned __int8 *)(a2 + 18);
      v7 = v6 & 0x7FFFFF;
      v8 = v10 & 0x800000;
    }
    else
    {
      v3 = *(unsigned __int8 *)(a2 + 18);
      v7 = v6 & 0x7FFFFF;
      v8 = (v5 << 23) & 0x7F800000;
    }
  }
  *(_DWORD *)(a1 + 8) = 32;
  LOBYTE(v3) = (unsigned __int8)v3 >> 3;
  *(_QWORD *)a1 = v7 | v8 | (unsigned int)(v3 << 31);
  return a1;
}
