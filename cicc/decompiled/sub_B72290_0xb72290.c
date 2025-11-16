// Function: sub_B72290
// Address: 0xb72290
//
__int64 __fastcall sub_B72290(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdi

  v2 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = v3 + 16 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
        {
          v5 = *(_QWORD *)(v3 + 8);
          if ( v5 )
            break;
        }
        v3 += 16;
        if ( v4 == v3 )
          goto LABEL_10;
      }
      if ( *(_DWORD *)(v5 + 32) > 0x40u )
      {
        v6 = *(_QWORD *)(v5 + 24);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      v3 += 16;
      sub_BD7260(v5);
      sub_BD2DD0(v5);
    }
    while ( v4 != v3 );
LABEL_10:
    v2 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16 * v2, 8);
}
