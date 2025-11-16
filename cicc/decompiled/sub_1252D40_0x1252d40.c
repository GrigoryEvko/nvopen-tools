// Function: sub_1252D40
// Address: 0x1252d40
//
__int64 __fastcall sub_1252D40(__int64 a1, __int64 a2, size_t a3)
{
  unsigned __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // r12
  unsigned __int64 v7; // rdx
  size_t v8; // rbx
  unsigned __int64 v9; // rdx

  v4 = *(_QWORD *)(a1 + 56);
  result = 0x4FCACE213F2B3885LL * v4;
  if ( 0x4FCACE213F2B3885LL * v4 <= 0x3531DEC0D4C77B0LL )
  {
    result = (__int64)sub_1252CA0(*(_QWORD *)(a1 + 48), *(_BYTE *)(a1 + 68), v4, 2 * (*(_BYTE *)(a1 + 69) == 0));
    *(_BYTE *)(a1 + 69) = 0;
  }
  if ( a3 )
  {
    v6 = 0;
    while ( 1 )
    {
      v7 = *(_QWORD *)(a1 + 56) % 0x4DuLL;
      if ( !v7 )
        v7 = 77;
      v8 = v7;
      if ( v7 > a3 )
        v8 = a3;
      result = sub_CB6200(*(_QWORD *)(a1 + 48), (unsigned __int8 *)(a2 + v6), v8);
      v9 = *(_QWORD *)(a1 + 56) - v8;
      *(_QWORD *)(a1 + 56) = v9;
      a3 -= v8;
      if ( !a3 )
        break;
      v6 += v8;
      sub_1252CA0(*(_QWORD *)(a1 + 48), *(_BYTE *)(a1 + 68), v9, 2);
    }
  }
  return result;
}
