// Function: sub_2F424E0
// Address: 0x2f424e0
//
__int64 __fastcall sub_2F424E0(__int64 a1, _BYTE *a2, unsigned int a3)
{
  unsigned int v3; // r9d
  __int64 v5; // rcx
  unsigned int v6; // r14d
  __int16 *v7; // r13
  unsigned int *v8; // rax
  unsigned int v9; // r8d
  __int64 v10; // rcx
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rbx
  _BYTE *v14; // rax
  int v15; // edx

  v3 = 0;
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 24LL * a3 + 16) & 0xFFF;
  v7 = (__int16 *)(*(_QWORD *)(v5 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v5 + 8) + 24LL * a3 + 16) >> 12));
  do
  {
    if ( !v7 )
      break;
    v8 = (unsigned int *)(*(_QWORD *)(a1 + 808) + 4LL * v6);
    v9 = *v8;
    if ( *v8 )
    {
      if ( v9 == 1 )
      {
        *v8 = 0;
        v3 = 1;
      }
      else
      {
        v10 = *(unsigned int *)(a1 + 424);
        v11 = *(_QWORD *)(a1 + 416);
        v12 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 624) + 2LL * (v9 & 0x7FFFFFFF));
        if ( v12 < (unsigned int)v10 )
        {
          while ( 1 )
          {
            v13 = v11 + 24LL * v12;
            if ( (v9 & 0x7FFFFFFF) == (*(_DWORD *)(v13 + 8) & 0x7FFFFFFF) )
              break;
            v12 += 0x10000;
            if ( (unsigned int)v10 <= v12 )
              goto LABEL_16;
          }
        }
        else
        {
LABEL_16:
          v13 = v11 + 24 * v10;
        }
        v14 = a2;
        if ( (*a2 & 4) == 0 && (a2[44] & 8) != 0 )
        {
          do
            v14 = (_BYTE *)*((_QWORD *)v14 + 1);
          while ( (v14[44] & 8) != 0 );
        }
        sub_2F41820(a1, *((_QWORD *)v14 + 1), v9, *(_WORD *)(v13 + 12));
        sub_2F42240(a1, *(unsigned __int16 *)(v13 + 12), 0);
        *(_BYTE *)(v13 + 15) = 1;
        *(_WORD *)(v13 + 12) = 0;
      }
    }
    v15 = *v7++;
    v6 += v15;
  }
  while ( (_WORD)v15 );
  return v3;
}
