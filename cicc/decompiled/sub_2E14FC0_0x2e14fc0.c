// Function: sub_2E14FC0
// Address: 0x2e14fc0
//
void __fastcall sub_2E14FC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rdx
  __int64 v6; // rbx
  unsigned __int64 v7; // r13
  __int64 *v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rsi

  v5 = (__int64 *)sub_2E09D00((__int64 *)a2, a3);
  if ( v5 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
    && (*(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v5 >> 1) & 3) <= (*(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                         | (unsigned int)(a3 >> 1) & 3) )
  {
    v10 = v5[2];
    if ( v10 )
      sub_2E0A600(a2, v10);
  }
  v6 = *(_QWORD *)(a2 + 104);
  if ( v6 )
  {
    v7 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    do
    {
      while ( 1 )
      {
        v8 = (__int64 *)sub_2E09D00((__int64 *)v6, a3);
        if ( v8 != (__int64 *)(*(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8))
          && (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3) <= ((unsigned int)(a3 >> 1)
                                                                                               & 3
                                                                                               | *(_DWORD *)(v7 + 24)) )
        {
          v9 = v8[2];
          if ( v9 )
          {
            if ( (*(_QWORD *)(v9 + 8) & 0xFFFFFFFFFFFFFFF8LL) == v7 )
              break;
          }
        }
        v6 = *(_QWORD *)(v6 + 104);
        if ( !v6 )
          goto LABEL_11;
      }
      sub_2E0A600(v6, v9);
      v6 = *(_QWORD *)(v6 + 104);
    }
    while ( v6 );
  }
LABEL_11:
  sub_2E0AF60(a2);
}
