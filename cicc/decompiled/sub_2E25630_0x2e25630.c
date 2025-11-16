// Function: sub_2E25630
// Address: 0x2e25630
//
void __fastcall sub_2E25630(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // ecx
  __int64 v6; // rdx
  int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rax

LABEL_1:
  v5 = *(_DWORD *)(a3 + 8);
  while ( v5 )
  {
    v6 = v5--;
    v9 = *(unsigned int *)(*(_QWORD *)a3 + 4 * v6 - 4);
    *(_DWORD *)(a3 + 8) = v5;
    v7 = v9;
    v8 = *(_QWORD *)(a1[12] + 56LL) + 2LL * *(unsigned int *)(*(_QWORD *)(a1[12] + 8LL) + 24 * v9 + 4);
    v9 = (unsigned __int16)v9;
    if ( v8 )
    {
      while ( 1 )
      {
        v8 += 2;
        *(_QWORD *)(a1[13] + 8 * v9) = a2;
        *(_QWORD *)(a1[16] + 8 * v9) = 0;
        if ( !*(_WORD *)(v8 - 2) )
          break;
        v7 += *(__int16 *)(v8 - 2);
        v9 = (unsigned __int16)v7;
      }
      goto LABEL_1;
    }
  }
}
