// Function: sub_3163050
// Address: 0x3163050
//
__int64 __fastcall sub_3163050(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rbx
  __int64 v4; // r12
  bool v5; // cc
  unsigned __int64 v6; // rdi

  v2 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = v3 + 32 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v3 != -8192 && *(_QWORD *)v3 != -4096 )
        {
          if ( *(_BYTE *)(v3 + 24) )
          {
            v5 = *(_DWORD *)(v3 + 16) <= 0x40u;
            *(_BYTE *)(v3 + 24) = 0;
            if ( !v5 )
            {
              v6 = *(_QWORD *)(v3 + 8);
              if ( v6 )
                break;
            }
          }
        }
        v3 += 32;
        if ( v4 == v3 )
          goto LABEL_10;
      }
      j_j___libc_free_0_0(v6);
      v3 += 32;
    }
    while ( v4 != v3 );
LABEL_10:
    v2 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 32 * v2, 8);
}
