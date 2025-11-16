// Function: sub_877D80
// Address: 0x877d80
//
__int64 __fastcall sub_877D80(__int64 a1, __int64 *a2)
{
  __int64 v4; // rsi
  __m128i *v5; // rdi
  int v6; // edx
  __int64 result; // rax
  _DWORD v8[5]; // [rsp+Ch] [rbp-14h] BYREF

  *(_QWORD *)a1 = a2;
  v4 = *a2;
  v8[0] = 0;
  if ( v4 != qword_4F600E0 )
    sub_877D50(a1, v4);
  if ( !*(_DWORD *)(a1 + 64) )
  {
    v5 = *(__m128i **)(a1 + 72);
    *(_QWORD *)(a1 + 64) = a2[6];
    if ( v5 )
    {
      sub_727480(v5);
    }
    else if ( *((_DWORD *)a2 + 12) )
    {
      *(_QWORD *)(a1 + 72) = sub_7274B0(*(_BYTE *)(a1 - 8) & 1);
    }
  }
  *(_BYTE *)(a1 + 88) &= ~4u;
  sub_85EBD0((__int64)a2, v8);
  v6 = v8[0] & 1;
  result = v6 | *(_BYTE *)(a1 + 89) & 0xFEu;
  *(_BYTE *)(a1 + 89) = v6 | *(_BYTE *)(a1 + 89) & 0xFE;
  return result;
}
