// Function: sub_287D4E0
// Address: 0x287d4e0
//
__int64 __fastcall sub_287D4E0(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int8 v9; // al
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 i; // rax
  _BYTE *v14; // rdx
  unsigned __int8 v15; // al
  __int64 *v16; // rdx
  const void *v17; // rax
  size_t v18; // rdx

  v7 = sub_D49300(a1, (__int64)a2, a3, a4, a5, a6);
  if ( !v7 )
    return 0;
  v8 = v7;
  v9 = *(_BYTE *)(v7 - 16);
  v10 = (v9 & 2) != 0 ? *(unsigned int *)(v8 - 24) : (*(_WORD *)(v8 - 16) >> 6) & 0xFu;
  if ( (unsigned int)v10 <= 1 )
    return 0;
  v11 = 8 * v10;
  v12 = 8;
  if ( (v9 & 2) == 0 )
    goto LABEL_19;
LABEL_6:
  for ( i = *(_QWORD *)(v8 - 32); ; i = v8 + -16 - 8LL * ((v9 >> 2) & 0xF) )
  {
    v14 = *(_BYTE **)(i + v12);
    if ( (unsigned __int8)(*v14 - 5) <= 0x1Fu )
    {
      v15 = *(v14 - 16);
      v16 = (v15 & 2) != 0 ? (__int64 *)*((_QWORD *)v14 - 4) : (__int64 *)&v14[-16 - 8LL * ((v15 >> 2) & 0xF)];
      if ( !*(_BYTE *)*v16 )
      {
        v17 = (const void *)sub_B91420(*v16);
        if ( a3 <= v18 && (!a3 || !memcmp(v17, a2, a3)) )
          break;
      }
    }
    v12 += 8;
    if ( v11 == v12 )
      return 0;
    v9 = *(_BYTE *)(v8 - 16);
    if ( (v9 & 2) != 0 )
      goto LABEL_6;
LABEL_19:
    ;
  }
  return 1;
}
