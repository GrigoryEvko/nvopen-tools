// Function: sub_2A11940
// Address: 0x2a11940
//
_BYTE *__fastcall sub_2A11940(__int64 a1, const void *a2, size_t a3)
{
  unsigned __int8 v4; // al
  __int64 v5; // rbx
  __int64 v6; // rdx
  _QWORD *v7; // r13
  _QWORD *v8; // rbx
  __int64 *v9; // rax
  const void *v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // r15
  unsigned __int8 v13; // al

  v4 = *(_BYTE *)(a1 - 16);
  if ( (v4 & 2) != 0 )
  {
    v5 = *(_QWORD *)(a1 - 32);
    v6 = *(unsigned int *)(a1 - 24);
  }
  else
  {
    v6 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
    v5 = a1 - 8LL * ((v4 >> 2) & 0xF) - 16;
  }
  v7 = (_QWORD *)(v5 + 8 * v6);
  v8 = (_QWORD *)(v5 + 8);
  if ( v7 == v8 )
    return 0;
  while ( 1 )
  {
    v12 = (_BYTE *)*v8;
    if ( (unsigned __int8)(*(_BYTE *)*v8 - 5) <= 0x1Fu )
    {
      v13 = *(v12 - 16);
      v9 = (v13 & 2) != 0 ? (__int64 *)*((_QWORD *)v12 - 4) : (__int64 *)&v12[-16 - 8LL * ((v13 >> 2) & 0xF)];
      if ( !*(_BYTE *)*v9 )
      {
        v10 = (const void *)sub_B91420(*v9);
        if ( a3 == v11 && (!a3 || !memcmp(a2, v10, a3)) )
          break;
      }
    }
    if ( v7 == ++v8 )
      return 0;
  }
  return v12;
}
