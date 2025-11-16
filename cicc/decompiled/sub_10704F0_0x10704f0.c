// Function: sub_10704F0
// Address: 0x10704f0
//
char __fastcall sub_10704F0(__int64 a1, __int64 a2)
{
  size_t *v2; // rsi
  size_t v3; // rbx
  size_t v4; // r12
  const void *v5; // rsi
  int v6; // eax
  size_t *v7; // rdi
  const void *v8; // rdi
  size_t v9; // rdx
  unsigned int v10; // eax

  if ( (*(_BYTE *)(*(_QWORD *)a2 + 8LL) & 1) != 0 )
  {
    v2 = *(size_t **)(*(_QWORD *)a2 - 8LL);
    v3 = 0;
    v4 = *v2;
    v5 = v2 + 3;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 8LL) & 1) != 0 )
    {
      v7 = *(size_t **)(*(_QWORD *)a1 - 8LL);
      v3 = *v7;
      v8 = v7 + 3;
      v9 = v3;
      if ( v4 <= v3 )
        v9 = v4;
      if ( v9 )
      {
        v10 = memcmp(v8, v5, v9);
        if ( v10 )
          return v10 >> 31;
      }
    }
  }
  else
  {
    LOBYTE(v6) = 0;
    if ( (*(_BYTE *)(*(_QWORD *)a1 + 8LL) & 1) == 0 )
      return v6;
    v4 = 0;
    v3 = **(_QWORD **)(*(_QWORD *)a1 - 8LL);
  }
  LOBYTE(v6) = v3 != v4 && v3 < v4;
  return v6;
}
