// Function: sub_24F9770
// Address: 0x24f9770
//
__int64 __fastcall sub_24F9770(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  __int64 v6; // r15
  _QWORD *v7; // rbx
  _QWORD *v8; // rax
  unsigned int v9; // r8d
  unsigned __int64 v10; // rsi
  _QWORD v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v12[0] = a2;
  v5 = sub_24F9690((__int64)a1, v12);
  v6 = *a1;
  v7 = v5;
  v12[0] = a3;
  v8 = sub_24F9690((__int64)a1, v12);
  v9 = 1;
  v10 = a1[34]
      + 8
      * ((((__int64)v8 - v6) >> 3)
       + 2 * ((((__int64)v8 - v6) >> 3) + (((unsigned __int64)v8 - v6) & 0xFFFFFFFFFFFFFFF8LL)));
  if ( (*(_QWORD *)(*(_QWORD *)(v10 + 72) + 8LL * ((unsigned int)(((__int64)v7 - v6) >> 3) >> 6))
      & (1LL << (((__int64)v7 - v6) >> 3))) == 0 )
  {
    v9 = 0;
    if ( a2 == a3 )
      return *(unsigned __int8 *)(v10 + 146);
  }
  return v9;
}
