// Function: sub_134B7F0
// Address: 0x134b7f0
//
__int64 __fastcall sub_134B7F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned int v7; // eax
  char v8; // cl
  unsigned int v9; // eax
  unsigned __int64 v10; // r14
  int v11; // r15d
  __int64 v12; // r13
  __int64 *v13; // r13

  result = *(_QWORD *)(a2 + 104);
  *(_BYTE *)(a2 + 18) = 1;
  if ( result )
  {
    if ( result != 512 )
    {
      v4 = sub_130F9A0(*(_QWORD *)(a2 + 96) << 12);
      if ( v4 > 0x7000000000000000LL )
      {
        v12 = 3184;
        v10 = 199;
        LOBYTE(v11) = -57;
      }
      else
      {
        v5 = v4 - 1;
        _BitScanReverse64(&v6, v4);
        v7 = v6 - ((((v4 - 1) & v4) == 0) - 1);
        if ( v7 < 0xE )
          v7 = 14;
        v8 = v7 - 3;
        v9 = v7 - 14;
        if ( !v9 )
          v8 = 12;
        v10 = ((v5 >> v8) & 3) + 4 * v9;
        v11 = ((v5 >> v8) & 3) + 4 * v9;
        v12 = 16 * v10;
      }
      v13 = (__int64 *)(a1 + v12);
      if ( sub_1348AA0(v13) )
        *(_QWORD *)(a1 + 8 * (v10 >> 6) + 1024) |= 1LL << v11;
      return sub_1348DA0(v13, (_QWORD *)a2);
    }
  }
  else
  {
    *(_QWORD *)(a2 + 40) = a2;
    *(_QWORD *)(a2 + 48) = a2;
    result = *(_QWORD *)(a1 + 4224);
    if ( result )
    {
      *(_QWORD *)(a2 + 40) = *(_QWORD *)(result + 48);
      *(_QWORD *)(*(_QWORD *)(a1 + 4224) + 48LL) = a2;
      *(_QWORD *)(a2 + 48) = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 40LL);
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 4224) + 48LL) + 40LL) = *(_QWORD *)(a1 + 4224);
      result = *(_QWORD *)(a2 + 48);
      *(_QWORD *)(result + 40) = a2;
    }
    *(_QWORD *)(a1 + 4224) = a2;
  }
  return result;
}
