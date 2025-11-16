// Function: sub_A630D0
// Address: 0xa630d0
//
__int64 __fastcall sub_A630D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // rdi
  int v9; // edx
  int v10; // r9d

  if ( *(_QWORD *)(a1 + 8) )
  {
    sub_A5A2A0(a1);
    *(_QWORD *)(a1 + 8) = 0;
  }
  if ( *(_QWORD *)(a1 + 16) )
  {
    if ( !*(_BYTE *)(a1 + 24) )
      sub_A5A0C0(a1);
  }
  v3 = *(unsigned int *)(a1 + 208);
  v4 = *(_QWORD *)(a1 + 192);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_8:
      if ( v6 != (__int64 *)(v4 + 16 * v3) )
        return *((unsigned int *)v6 + 2);
    }
    else
    {
      v9 = 1;
      while ( v7 != -4096 )
      {
        v10 = v9 + 1;
        v5 = (v3 - 1) & (v9 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_8;
        v9 = v10;
      }
    }
  }
  return 0xFFFFFFFFLL;
}
