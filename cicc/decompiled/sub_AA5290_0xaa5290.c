// Function: sub_AA5290
// Address: 0xaa5290
//
__int64 __fastcall sub_AA5290(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // rdi
  _QWORD *v4; // r15
  _QWORD *v5; // r12
  unsigned __int64 *v6; // rcx
  unsigned __int64 v7; // rdx
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  unsigned __int64 *v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 i; // rax
  __int64 v17; // r12
  __int64 v18; // rsi

  if ( (*(_WORD *)(a1 + 2) & 0x7FFF) != 0 )
  {
    v13 = sub_AA48A0(a1);
    v14 = sub_BCB2D0(v13);
    v15 = sub_ACD640(v14, 1, 0);
    for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(a1 + 16) )
    {
      v17 = *(_QWORD *)(i + 24);
      v18 = sub_AD4C70(v15, *(_QWORD *)(v17 + 8), 0);
      sub_BD84D0(v17, v18);
      sub_ACFDF0(v17);
    }
  }
  sub_AA5200(a1);
  v1 = *(_QWORD *)(a1 + 56);
  v2 = a1 + 48;
  if ( a1 + 48 != v1 )
  {
    do
    {
      if ( !v1 )
        BUG();
      v3 = *(_QWORD *)(v1 + 40);
      if ( v3 )
        sub_B14200(v3);
      v1 = *(_QWORD *)(v1 + 8);
    }
    while ( v2 != v1 );
    v4 = *(_QWORD **)(a1 + 56);
    if ( (_QWORD *)v2 != v4 )
    {
      do
      {
        v5 = v4;
        v4 = (_QWORD *)v4[1];
        sub_AA4910(v1, (__int64)(v5 - 3));
        v6 = (unsigned __int64 *)v5[1];
        v7 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
        *v6 = v7 | *v6 & 7;
        *(_QWORD *)(v7 + 8) = v6;
        *v5 &= 7uLL;
        v5[1] = 0;
        sub_BD72D0(v5 - 3);
      }
      while ( (_QWORD *)v1 != v4 );
      v8 = *(_QWORD **)(a1 + 56);
      while ( (_QWORD *)v2 != v8 )
      {
        v9 = v8;
        v8 = (_QWORD *)v8[1];
        sub_AA4910(v2, (__int64)(v9 - 3));
        v10 = (unsigned __int64 *)v9[1];
        v11 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
        *v10 = v11 | *v10 & 7;
        *(_QWORD *)(v11 + 8) = v10;
        *v9 &= 7uLL;
        v9[1] = 0;
        sub_BD72D0(v9 - 3);
      }
    }
  }
  return sub_BD7260(a1);
}
