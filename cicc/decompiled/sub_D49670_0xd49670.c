// Function: sub_D49670
// Address: 0xd49670
//
_BYTE *__fastcall sub_D49670(__int64 a1, const void *a2, size_t a3)
{
  unsigned __int8 v3; // al
  __int64 v6; // rsi
  __int64 v7; // r8
  _QWORD *v8; // rbx
  _QWORD *v9; // rdx
  _QWORD *i; // r14
  __int64 *v11; // rax
  const void *v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // r15
  unsigned __int8 v15; // al

  if ( a1 )
  {
    v3 = *(_BYTE *)(a1 - 16);
    if ( (v3 & 2) != 0 )
    {
      v7 = *(_QWORD *)(a1 - 32);
      v6 = *(unsigned int *)(a1 - 24);
    }
    else
    {
      v6 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
      v7 = a1 - 8LL * ((v3 >> 2) & 0xF) - 16;
    }
    v8 = (_QWORD *)sub_D46550(v7, v6, 1);
    for ( i = v9; i != v8; ++v8 )
    {
      v14 = (_BYTE *)*v8;
      if ( (unsigned __int8)(*(_BYTE *)*v8 - 5) > 0x1Fu )
        continue;
      v15 = *(v14 - 16);
      if ( (v15 & 2) != 0 )
      {
        if ( !*((_DWORD *)v14 - 6) )
          continue;
        v11 = (__int64 *)*((_QWORD *)v14 - 4);
      }
      else
      {
        if ( (*((_WORD *)v14 - 8) & 0x3C0) == 0 )
          continue;
        v11 = (__int64 *)&v14[-16 - 8LL * ((v15 >> 2) & 0xF)];
      }
      if ( !*(_BYTE *)*v11 )
      {
        v12 = (const void *)sub_B91420(*v11);
        if ( a3 == v13 && (!a3 || !memcmp(a2, v12, a3)) )
          return v14;
      }
    }
  }
  return 0;
}
