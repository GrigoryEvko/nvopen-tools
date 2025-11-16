// Function: sub_2054800
// Address: 0x2054800
//
__int64 __fastcall sub_2054800(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned int v4; // r14d
  __int64 v5; // rbx
  unsigned int v6; // r13d

  v3 = a3 + 40;
  if ( a3 + 40 != a2 )
  {
    v4 = *(_DWORD *)(a1 + 32);
    v5 = a2;
    v6 = 0;
    while ( 1 )
    {
      while ( v4 != *(_DWORD *)(v5 + 32) )
      {
        if ( v4 < *(_DWORD *)(v5 + 32) )
          goto LABEL_4;
LABEL_5:
        v5 += 40;
        if ( v3 == v5 )
          return v6;
      }
      if ( (int)sub_16AEA10(*(_QWORD *)(v5 + 8) + 24LL, *(_QWORD *)(a1 + 8) + 24LL) < 0 )
      {
LABEL_4:
        ++v6;
        goto LABEL_5;
      }
      v5 += 40;
      if ( v3 == v5 )
        return v6;
    }
  }
  return 0;
}
