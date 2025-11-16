// Function: sub_3143E70
// Address: 0x3143e70
//
__int64 __fastcall sub_3143E70(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // dl
  __int64 v4; // rsi
  unsigned int v5; // edx
  int v6; // ecx

  result = a1;
  if ( a2
    && ((v3 = *(_BYTE *)(a2 - 16), (v3 & 2) != 0)
      ? (v4 = *(_QWORD *)(a2 - 32))
      : (v4 = a2 - 16 - 8LL * ((v3 >> 2) & 0xF)),
        **(_BYTE **)v4 == 20 && (v5 = *(_DWORD *)(*(_QWORD *)v4 + 4LL), (v5 & 7) == 7) && (v5 & 0xFFFFFFF8) != 0) )
  {
    *(_DWORD *)(a1 + 12) = 0;
    *(_BYTE *)(a1 + 20) = 1;
    v6 = (unsigned __int16)(v5 >> 3);
    if ( (v5 & 0x10000000) != 0 )
      v6 = (unsigned __int16)v5 >> 3;
    *(_DWORD *)a1 = v6;
    *(_DWORD *)(a1 + 4) = (v5 >> 26) & 3;
    *(_DWORD *)(a1 + 8) = v5 >> 29;
    *(float *)(a1 + 16) = (float)((v5 >> 19) & 0x7F) / 100.0;
  }
  else
  {
    *(_BYTE *)(a1 + 20) = 0;
  }
  return result;
}
