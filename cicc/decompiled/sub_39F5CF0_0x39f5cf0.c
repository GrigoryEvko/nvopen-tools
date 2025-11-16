// Function: sub_39F5CF0
// Address: 0x39f5cf0
//
char *__fastcall sub_39F5CF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char *v4; // rdx
  char *v5; // rsi
  __int64 v6; // rcx
  char *result; // rax
  _QWORD *v8; // rax
  __int64 v9; // [rsp+8h] [rbp-10h] BYREF

  if ( ((*(_BYTE *)(a2 + 199) & 0x40) == 0 || !*(_BYTE *)(a2 + 223)) && !*(_QWORD *)(a2 + 56) )
  {
    if ( byte_5057707 != 8 )
      goto LABEL_33;
    v9 = *(_QWORD *)(a2 + 144);
    if ( (*(_BYTE *)(a2 + 199) & 0x40) != 0 )
      *(_BYTE *)(a2 + 223) = 0;
    *(_QWORD *)(a2 + 56) = &v9;
  }
  v3 = 0;
  do
  {
    while ( 1 )
    {
      v4 = *(char **)(a1 + 8 * v3);
      v5 = *(char **)(a2 + 8 * v3);
      if ( *(_BYTE *)(a1 + v3 + 216) )
        goto LABEL_33;
      if ( *(_BYTE *)(a2 + v3 + 216) )
      {
        if ( v4 )
        {
          if ( byte_5057700[v3] != 8 )
            goto LABEL_33;
          *(_QWORD *)v4 = v5;
        }
        goto LABEL_8;
      }
      if ( v4 != 0 && v5 != 0 && v4 != v5 )
        break;
LABEL_8:
      if ( ++v3 == 17 )
        goto LABEL_19;
    }
    v6 = (unsigned __int8)byte_5057700[v3];
    if ( (unsigned int)v6 < 8 )
    {
      if ( (v6 & 4) != 0 )
      {
        *(_DWORD *)v4 = *(_DWORD *)v5;
        *(_DWORD *)&v4[v6 - 4] = *(_DWORD *)&v5[v6 - 4];
      }
      else if ( byte_5057700[v3] )
      {
        *v4 = *v5;
        if ( (v6 & 2) != 0 )
          *(_WORD *)&v4[v6 - 2] = *(_WORD *)&v5[v6 - 2];
      }
      goto LABEL_8;
    }
    ++v3;
    *(_QWORD *)v4 = *(_QWORD *)v5;
    *(_QWORD *)&v4[v6 - 8] = *(_QWORD *)&v5[v6 - 8];
    qmemcpy(
      (void *)((unsigned __int64)(v4 + 8) & 0xFFFFFFFFFFFFFFF8LL),
      (const void *)(v5 - &v4[-((unsigned __int64)(v4 + 8) & 0xFFFFFFFFFFFFFFF8LL)]),
      8LL * (((unsigned int)v4 - (((_DWORD)v4 + 8) & 0xFFFFFFF8) + (unsigned int)v6) >> 3));
  }
  while ( v3 != 17 );
LABEL_19:
  result = 0;
  if ( ((*(_BYTE *)(a1 + 199) & 0x40) == 0 || !*(_BYTE *)(a1 + 223)) && !*(_QWORD *)(a1 + 56) )
  {
    v8 = *(_QWORD **)(a2 + 56);
    if ( (*(_BYTE *)(a2 + 199) & 0x40) != 0 && *(_BYTE *)(a2 + 223) )
      return (char *)v8 + *(_QWORD *)(a2 + 208) - *(_QWORD *)(a1 + 144);
    if ( byte_5057707 == 8 )
    {
      v8 = (_QWORD *)*v8;
      return (char *)v8 + *(_QWORD *)(a2 + 208) - *(_QWORD *)(a1 + 144);
    }
LABEL_33:
    abort();
  }
  return result;
}
