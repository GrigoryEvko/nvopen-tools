// Function: sub_134B500
// Address: 0x134b500
//
__int64 __fastcall sub_134B500(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdx
  unsigned __int64 v6; // rsi

  result = sub_134B450(a2);
  v3 = result + 528;
  if ( a2 == *(_QWORD *)(a1 + 8 * result + 4232) )
  {
    v5 = *(_QWORD *)(a2 + 64);
    if ( a2 == v5 )
    {
      *(_QWORD *)(a1 + 8 * v3 + 8) = 0;
      goto LABEL_6;
    }
    *(_QWORD *)(a1 + 8 * v3 + 8) = v5;
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 72) + 64LL) = *(_QWORD *)(*(_QWORD *)(a2 + 64) + 72LL);
  v4 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(*(_QWORD *)(a2 + 64) + 72LL) = v4;
  *(_QWORD *)(a2 + 72) = *(_QWORD *)(v4 + 64);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 64) + 72LL) + 64LL) = *(_QWORD *)(a2 + 64);
  *(_QWORD *)(*(_QWORD *)(a2 + 72) + 64LL) = a2;
  if ( !*(_QWORD *)(a1 + 8 * v3 + 8) )
  {
LABEL_6:
    v6 = result;
    result &= 0x3Fu;
    *(_QWORD *)(a1 + 8 * (v6 >> 6) + 5256) &= __ROL8__(-2, result);
  }
  return result;
}
