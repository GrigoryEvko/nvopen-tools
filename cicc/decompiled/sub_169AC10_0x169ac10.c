// Function: sub_169AC10
// Address: 0x169ac10
//
__int64 __fastcall sub_169AC10(__int64 a1, __int64 a2)
{
  char v3; // al
  char v4; // cl
  int v5; // ebx
  __int64 v6; // rsi
  int v7; // esi
  int v8; // edx
  __int16 v10; // dx
  __int64 v11; // rdx
  __int16 v12; // si

  v3 = *(_BYTE *)(a2 + 18);
  v4 = v3 & 7;
  if ( (v3 & 7) == 1 )
  {
    v11 = *(_QWORD *)sub_16984A0(a2);
    v3 = *(_BYTE *)(a2 + 18);
    v12 = v11;
    v8 = 31744;
    v7 = v12 & 0x3FF;
  }
  else if ( v4 == 3 || !v4 )
  {
    v7 = 0;
    v8 = 0;
    if ( v4 != 3 )
      v8 = 31744;
  }
  else
  {
    v5 = *(__int16 *)(a2 + 16) + 15;
    v6 = *(_QWORD *)sub_16984A0(a2);
    if ( v5 == 1 )
    {
      v10 = v6;
      v3 = *(_BYTE *)(a2 + 18);
      v7 = v6 & 0x3FF;
      v8 = v10 & 0x400;
    }
    else
    {
      v3 = *(_BYTE *)(a2 + 18);
      v7 = v6 & 0x3FF;
      v8 = ((_WORD)v5 << 10) & 0x7C00;
    }
  }
  *(_DWORD *)(a1 + 8) = 16;
  *(_QWORD *)a1 = v7 | v8 | ((unsigned __int8)((v3 & 8) != 0) << 15);
  return a1;
}
