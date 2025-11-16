// Function: sub_1C6EAF0
// Address: 0x1c6eaf0
//
__int64 __fastcall sub_1C6EAF0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  int v4; // r12d
  unsigned __int64 v5; // r15
  _QWORD *v6; // rax
  _QWORD *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  _QWORD v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v11[0] = *(_QWORD *)(a2 + 112);
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, a2);
    v2 = *(_QWORD *)(a2 + 88);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      sub_15E08E0(a2, a2);
    v3 = *(_QWORD *)(a2 + 88);
  }
  else
  {
    v2 = *(_QWORD *)(a2 + 88);
    v3 = v2;
  }
  v4 = 2;
  v5 = v3 + 40LL * *(_QWORD *)(a2 + 96);
  if ( v2 == v5 )
    return 0;
  while ( 1 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) == 15 && !(unsigned __int8)sub_1560260(v11, v4, 6) )
    {
      v6 = *(_QWORD **)(a1 + 72);
      if ( !v6 )
        return 1;
      v7 = (_QWORD *)(a1 + 64);
      do
      {
        while ( 1 )
        {
          v8 = v6[2];
          v9 = v6[3];
          if ( v6[4] >= v2 )
            break;
          v6 = (_QWORD *)v6[3];
          if ( !v9 )
            goto LABEL_13;
        }
        v7 = v6;
        v6 = (_QWORD *)v6[2];
      }
      while ( v8 );
LABEL_13:
      if ( (_QWORD *)(a1 + 64) == v7 || v7[4] > v2 )
        return 1;
    }
    v2 += 40LL;
    ++v4;
    if ( v2 == v5 )
      return 0;
  }
}
