// Function: sub_25CD250
// Address: 0x25cd250
//
void __fastcall sub_25CD250(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 *v2; // r14
  __int64 v3; // r15
  __int64 v4; // rdx
  int v5; // eax
  char v6; // cl
  __int64 v7; // r8
  int v8; // eax
  char v9; // [rsp+Fh] [rbp-31h]

  v1 = *(__int64 **)a1;
  v2 = *(__int64 **)(a1 + 56);
  if ( *(__int64 **)a1 != v2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    do
    {
      v4 = *v1;
      if ( !*(_BYTE *)(v3 + 336) || *(char *)(v4 + 12) < 0 )
      {
        switch ( *(_BYTE *)(v4 + 12) & 0xF )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            v5 = *(_DWORD *)(v4 + 8);
            if ( !v5 )
            {
              v4 = *(_QWORD *)(v4 + 64);
              v5 = *(_DWORD *)(v4 + 8);
            }
            if ( v5 == 1 )
            {
              v6 = *(_BYTE *)(v4 + 12);
              if ( *(_QWORD *)(a1 + 24) == 1
                || (v6 & 0xFu) - 7 > 1
                || (v7 = *(_QWORD *)(v4 + 32), v7 == *(_QWORD *)(a1 + 40))
                && (!v7
                 || (v9 = *(_BYTE *)(v4 + 12),
                     v8 = memcmp(*(const void **)(v4 + 24), *(const void **)(a1 + 32), *(_QWORD *)(v4 + 32)),
                     v6 = v9,
                     !v8)) )
              {
                if ( (v6 & 0x40) == 0 )
                  return;
              }
            }
            break;
          case 2:
          case 4:
          case 9:
          case 0xA:
            break;
          default:
            BUG();
        }
      }
      *(_QWORD *)a1 = ++v1;
    }
    while ( v1 != v2 );
  }
}
