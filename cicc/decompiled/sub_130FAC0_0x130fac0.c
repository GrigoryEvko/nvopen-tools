// Function: sub_130FAC0
// Address: 0x130fac0
//
unsigned __int64 __fastcall sub_130FAC0(__int64 a1, char a2)
{
  __int64 v3; // r10
  __int64 v4; // rdx
  int v5; // edi
  __int64 v6; // rax
  __int64 v7; // r11
  __int64 v8; // rcx
  _QWORD *v9; // rsi
  int v10; // ecx
  __int64 v11; // rax
  unsigned __int64 result; // rax
  unsigned int v13; // r9d
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r12
  _DWORD *v18; // rax
  unsigned __int64 v19; // rdx

  v3 = a1 + 6576;
  *(_QWORD *)&dword_50607C0 = (unsigned __int64)(a2 != 0) << 12;
  v4 = a1 + 80;
  v5 = 0;
  v6 = v4;
  do
  {
    if ( *(_BYTE *)(v6 + 12) )
    {
      v7 = v5++;
      qword_5060180[v7] = (1LL << *(_DWORD *)v6) + ((__int64)*(int *)(v6 + 8) << *(_DWORD *)(v6 + 4));
    }
    v6 += 28;
  }
  while ( v3 != v6 );
  if ( v5 <= 199 )
  {
    v8 = v5;
    do
    {
      ++v5;
      qword_5060180[v8] = *(_QWORD *)(a1 + 64) + 4096LL;
    }
    while ( v5 != 200 );
  }
  v9 = qword_505FA40;
  do
  {
    v10 = *(_DWORD *)(v4 + 4);
    v11 = *(int *)(v4 + 8);
    ++v9;
    v4 += 28;
    *(v9 - 1) = (1LL << *(_DWORD *)(v4 - 28)) + (v11 << v10);
  }
  while ( qword_5060180 != v9 );
  result = 0;
  v13 = 0;
  do
  {
    v14 = ((__int64)*(int *)(a1 + 28LL * v13 + 88) << *(_DWORD *)(a1 + 28LL * v13 + 84))
        + (1LL << *(_DWORD *)(a1 + 28LL * v13 + 80))
        + 7;
    v15 = v14 >> 3;
    if ( v14 > 0x1007 )
      v15 = 512;
    if ( result <= v15 )
    {
      v16 = 1 - result;
      v17 = 0x101010101010101LL * (unsigned __int8)v13;
      v18 = (_DWORD *)(result + 84281344);
      v19 = v15 + v16;
      if ( v19 < 8 )
      {
        if ( (v19 & 4) != 0 )
        {
          *v18 = v17;
          *(_DWORD *)((char *)v18 + v19 - 4) = v17;
        }
        else if ( v19 )
        {
          *(_BYTE *)v18 = v13;
          if ( (v19 & 2) != 0 )
            *(_WORD *)((char *)v18 + v19 - 2) = v17;
        }
      }
      else
      {
        *(_QWORD *)v18 = v17;
        *(_QWORD *)((char *)v18 + v19 - 8) = v17;
        memset64(
          (void *)((unsigned __int64)(v18 + 2) & 0xFFFFFFFFFFFFFFF8LL),
          v17,
          ((unsigned __int64)v18 + v19 - ((unsigned __int64)(v18 + 2) & 0xFFFFFFFFFFFFFFF8LL)) >> 3);
      }
      result = v15 + 1;
    }
    ++v13;
  }
  while ( v13 <= 0xE7 && result <= 0x200 );
  return result;
}
