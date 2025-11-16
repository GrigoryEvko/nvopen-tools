// Function: sub_1ADC5B0
// Address: 0x1adc5b0
//
__int64 __fastcall sub_1ADC5B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // esi
  int v6; // eax
  int v7; // eax
  _QWORD v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v5 = *(_DWORD *)(a1 + 24);
  v6 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v7 = v6 + 1;
  if ( 4 * v7 >= 3 * v5 )
  {
    v5 *= 2;
  }
  else if ( v5 - *(_DWORD *)(a1 + 20) - v7 > v5 >> 3 )
  {
    goto LABEL_3;
  }
  sub_12E48B0(a1, v5);
  sub_12E4800(a1, a2, v9);
  a3 = v9[0];
  v7 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v7;
  if ( *(_QWORD *)(a3 + 24) != -8 )
    --*(_DWORD *)(a1 + 20);
  return a3;
}
