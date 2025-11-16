// Function: sub_359BB30
// Address: 0x359bb30
//
void __fastcall sub_359BB30(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  int v4; // edx
  unsigned int v5; // r12d
  __int64 v6; // rsi
  __int64 v7[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 56);
  for ( v7[0] = v2; v7[0] != a1 + 48; v2 = v7[0] )
  {
    if ( *(_WORD *)(v2 + 68) && *(_WORD *)(v2 + 68) != 68 )
      break;
    v4 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
    if ( v4 != 1 )
    {
      v5 = 1;
      while ( 1 )
      {
        v6 = v5 + 1;
        if ( a2 == *(_QWORD *)(*(_QWORD *)(v2 + 32) + 40 * v6 + 24) )
          break;
        v5 += 2;
        if ( v5 == v4 )
          goto LABEL_10;
      }
      sub_2E8A650(v2, v6);
      sub_2E8A650(v2, v5);
    }
LABEL_10:
    sub_2FD79B0(v7);
  }
}
