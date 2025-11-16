// Function: sub_9ADCB0
// Address: 0x9adcb0
//
__int64 __fastcall sub_9ADCB0(__int64 a1, __int64 a2, unsigned __int8 *a3, char a4)
{
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 *v8; // rax

  v6 = *(_DWORD *)(*(_QWORD *)a2 + 8LL);
  *(_DWORD *)(a1 + 8) = v6;
  if ( v6 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    *(_DWORD *)(a1 + 24) = v6;
    sub_C43690(a1 + 16, 0, 0);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = 0;
  }
  sub_9AB8E0(a3, *(_QWORD *)(a2 + 8), (unsigned __int64 *)a1, **(_DWORD **)(a2 + 16) + 1, *(__m128i **)(a2 + 24));
  v7 = **(_QWORD **)(a2 + 32);
  if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
    v8 = *(__int64 **)(v7 - 8);
  else
    v8 = (__int64 *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
  sub_9974C0(a1, *v8, a3, a4, **(_DWORD **)(a2 + 16), *(__int64 **)(a2 + 24));
  return a1;
}
