// Function: sub_3598EB0
// Address: 0x3598eb0
//
__int64 __fastcall sub_3598EB0(__int64 a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rbx
  _QWORD *v4; // r12
  _QWORD *v5; // r15
  __int64 v6; // rbx
  _QWORD *v7; // r12
  unsigned __int64 *v8; // rcx
  unsigned __int64 v9; // rdx

  v2 = *(_QWORD **)(a1 + 48);
  v3 = v2[7];
  v4 = v2 + 6;
  if ( (_QWORD *)v3 != v2 + 6 )
  {
    do
    {
      while ( 1 )
      {
        sub_2FAD510(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v3);
        if ( !v3 )
          BUG();
        if ( (*(_BYTE *)v3 & 4) == 0 )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v4 == (_QWORD *)v3 )
          goto LABEL_7;
      }
      while ( (*(_BYTE *)(v3 + 44) & 8) != 0 )
        v3 = *(_QWORD *)(v3 + 8);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v4 != (_QWORD *)v3 );
LABEL_7:
    v2 = *(_QWORD **)(a1 + 48);
    v5 = (_QWORD *)v2[7];
    if ( v2 + 6 != v5 )
    {
      v6 = (__int64)(v2 + 5);
      do
      {
        v7 = v5;
        v5 = (_QWORD *)v5[1];
        sub_2E31080(v6, (__int64)v7);
        v8 = (unsigned __int64 *)v7[1];
        v9 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
        *v8 = v9 | *v8 & 7;
        *(_QWORD *)(v9 + 8) = v8;
        *v7 &= 7uLL;
        v7[1] = 0;
        sub_2E310F0(v6);
      }
      while ( v2 + 6 != v5 );
      v2 = *(_QWORD **)(a1 + 48);
    }
  }
  return sub_2E32710(v2);
}
