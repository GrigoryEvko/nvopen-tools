// Function: sub_134B650
// Address: 0x134b650
//
char __fastcall sub_134B650(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned int v7; // eax
  char v8; // cl
  unsigned int v9; // eax
  unsigned __int64 v10; // r14
  int v11; // r15d
  __int64 v12; // r13
  _QWORD *v13; // r13
  __int64 v14; // rax

  v3 = *(_QWORD *)(a2 + 104);
  *(_BYTE *)(a2 + 18) = 0;
  if ( v3 )
  {
    if ( v3 != 512 )
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
      v13 = (_QWORD *)(a1 + v12);
      sub_1348F60(v13, (_QWORD *)a2);
      LOBYTE(v3) = sub_1348AA0(v13);
      if ( (_BYTE)v3 )
      {
        v3 = ~(1LL << v11);
        *(_QWORD *)(a1 + 8 * (v10 >> 6) + 1024) &= v3;
      }
    }
  }
  else
  {
    if ( a2 == *(_QWORD *)(a1 + 4224) )
    {
      v3 = *(_QWORD *)(a2 + 40);
      if ( a2 == v3 )
      {
        *(_QWORD *)(a1 + 4224) = 0;
        return v3;
      }
      *(_QWORD *)(a1 + 4224) = v3;
    }
    *(_QWORD *)(*(_QWORD *)(a2 + 48) + 40LL) = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL);
    v14 = *(_QWORD *)(a2 + 48);
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL) = v14;
    *(_QWORD *)(a2 + 48) = *(_QWORD *)(v14 + 40);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL) + 40LL) = *(_QWORD *)(a2 + 40);
    v3 = *(_QWORD *)(a2 + 48);
    *(_QWORD *)(v3 + 40) = a2;
  }
  return v3;
}
