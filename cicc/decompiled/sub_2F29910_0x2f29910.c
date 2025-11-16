// Function: sub_2F29910
// Address: 0x2f29910
//
void __fastcall sub_2F29910(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  signed int v5; // r14d
  unsigned __int16 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rsi
  int v9; // r10d
  unsigned int i; // edx
  __int64 v11; // rdi
  unsigned int v12; // edx

  if ( *(_WORD *)(a2 + 68) == 20 )
  {
    v3 = *(_QWORD *)(a2 + 32);
    v5 = *(_DWORD *)(v3 + 48);
    v6 = (*(_DWORD *)(v3 + 40) >> 8) & 0xFFF;
    if ( v5 < 0 || (unsigned __int8)sub_2EBF3A0(*(_QWORD **)(a1 + 24), v5) )
    {
      v7 = *(unsigned int *)(a1 + 72);
      if ( (_DWORD)v7 )
      {
        v8 = *(_QWORD *)(a1 + 56);
        v9 = 1;
        for ( i = (v7 - 1)
                & (((0xBF58476D1CE4E5B9LL * ((37 * (unsigned int)v6) | ((unsigned __int64)(unsigned int)(37 * v5) << 32))) >> 31)
                 ^ (756364221 * v6)); ; i = (v7 - 1) & v12 )
        {
          v11 = v8 + 16LL * i;
          if ( v5 == *(_DWORD *)v11 && v6 == *(_DWORD *)(v11 + 4) )
            break;
          if ( *(_DWORD *)v11 == -1 && *(_DWORD *)(v11 + 4) == -1 )
            return;
          v12 = v9 + i;
          ++v9;
        }
        if ( v11 != v8 + 16 * v7 && a2 == *(_QWORD *)(v11 + 8) )
        {
          *(_QWORD *)v11 = 0xFFFFFFFEFFFFFFFELL;
          --*(_DWORD *)(a1 + 64);
          ++*(_DWORD *)(a1 + 68);
        }
      }
    }
  }
}
