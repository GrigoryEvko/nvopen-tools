// Function: sub_FF0720
// Address: 0xff0720
//
void __fastcall sub_FF0720(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // rcx
  unsigned int v5; // edi
  int v6; // r11d
  unsigned __int64 v7; // rax
  unsigned int i; // edx
  __int64 *v9; // r9
  __int64 v10; // r10
  unsigned int v11; // edx
  __int64 *v12; // rsi
  int v13; // r11d
  unsigned int j; // eax
  __int64 *v15; // rdx
  unsigned int v16; // eax
  int v17; // eax

  v3 = *(_DWORD *)(a1 + 56);
  v4 = *(_QWORD *)(a1 + 40);
  if ( v3 )
  {
    v5 = v3 - 1;
    v6 = 1;
    for ( i = (v3 - 1) & (969526130 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))); ; i = v5 & v11 )
    {
      v9 = (__int64 *)(v4 + 24LL * i);
      v10 = *v9;
      if ( a2 == *v9 && !*((_DWORD *)v9 + 2) )
        break;
      if ( v10 == -4096 && *((_DWORD *)v9 + 2) == -1 )
        return;
      v11 = v6 + i;
      ++v6;
    }
    v12 = (__int64 *)(v4 + 24LL * v3);
    if ( v9 != v12 )
    {
      v13 = 1;
      v7 = (unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32;
      for ( j = v5 & (((0xBF58476D1CE4E5B9LL * (v7 | 0x25)) >> 31) ^ (484763065 * (v7 | 0x25))); ; j = v5 & v16 )
      {
        v15 = (__int64 *)(v4 + 24LL * j);
        if ( v10 == *v15 && *((_DWORD *)v15 + 2) == 1 )
          break;
        if ( *v15 == -4096 && *((_DWORD *)v15 + 2) == -1 )
        {
          v15 = v12;
          break;
        }
        v16 = v13 + j;
        ++v13;
      }
      v17 = *((_DWORD *)v9 + 4);
      *((_DWORD *)v9 + 4) = *((_DWORD *)v15 + 4);
      *((_DWORD *)v15 + 4) = v17;
    }
  }
}
