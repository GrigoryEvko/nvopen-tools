// Function: sub_AA4B40
// Address: 0xaa4b40
//
void __fastcall sub_AA4B40(__int64 a1)
{
  __int64 v1; // r13
  __int64 *v2; // r15
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 i; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-40h]
  __int64 v12; // [rsp+8h] [rbp-38h]

  v1 = a1 + 48;
  v2 = *(__int64 **)(a1 + 56);
  *(_WORD *)(a1 + 2) &= ~0x8000u;
  *(_BYTE *)(a1 + 40) = 0;
  if ( v2 != (__int64 *)(a1 + 48) )
  {
    do
    {
      while ( 1 )
      {
        if ( !v2 )
          BUG();
        v3 = v2[5];
        v11 = v3;
        if ( v3 )
          break;
        v2 = (__int64 *)v2[1];
        if ( (__int64 *)v1 == v2 )
          return;
      }
      v4 = sub_B14240(v3);
      v12 = v5;
      for ( i = v4; i != v12; i = *(_QWORD *)(i + 8) )
      {
        v7 = sub_AA4B30(a1);
        v8 = sub_B13CF0(i, v7, 0);
        sub_AA48C0(v1, v8);
        v9 = *v2;
        v10 = *(_QWORD *)(v8 + 24);
        *(_QWORD *)(v8 + 32) = v2;
        v9 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v9 | v10 & 7;
        *(_QWORD *)(v9 + 8) = v8 + 24;
        *v2 = *v2 & 7 | (v8 + 24);
      }
      sub_B14200(v11);
      v2 = (__int64 *)v2[1];
    }
    while ( (__int64 *)v1 != v2 );
  }
}
