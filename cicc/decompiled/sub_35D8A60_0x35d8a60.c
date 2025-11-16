// Function: sub_35D8A60
// Address: 0x35d8a60
//
__int64 __fastcall sub_35D8A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  unsigned int v5; // r15d
  __int64 v6; // rbx
  unsigned int v7; // r13d

  v4 = a4 + 40;
  if ( a4 + 40 != a3 )
  {
    v5 = *(_DWORD *)(a2 + 32);
    v6 = a3;
    v7 = 0;
    while ( 1 )
    {
      while ( v5 != *(_DWORD *)(v6 + 32) )
      {
        if ( v5 < *(_DWORD *)(v6 + 32) )
          goto LABEL_4;
LABEL_5:
        v6 += 40;
        if ( v4 == v6 )
          return v7;
      }
      if ( (int)sub_C4C880(*(_QWORD *)(v6 + 8) + 24LL, *(_QWORD *)(a2 + 8) + 24LL) < 0 )
      {
LABEL_4:
        ++v7;
        goto LABEL_5;
      }
      v6 += 40;
      if ( v4 == v6 )
        return v7;
    }
  }
  return 0;
}
