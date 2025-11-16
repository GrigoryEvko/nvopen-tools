// Function: sub_36D0D50
// Address: 0x36d0d50
//
__int64 __fastcall sub_36D0D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 v9; // ax
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax

  v9 = *(_WORD *)(a1 + 68);
  if ( v9 > 0x1B57u )
  {
    a5 = 0;
    if ( (unsigned __int16)(v9 - 7006) > 1u )
      return (unsigned int)a5;
  }
  else if ( v9 <= 0x1B55u )
  {
    if ( (unsigned __int16)(v9 - 2659) <= 0x13u && ((1LL << ((unsigned __int8)v9 - 99)) & 0xFFDFB) != 0 )
    {
      v10 = *(unsigned int *)(a4 + 8);
      if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v10 + 1, 8u, a5, a6);
        v10 = *(unsigned int *)(a4 + 8);
      }
      LODWORD(a5) = 1;
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v10) = a1;
      ++*(_DWORD *)(a4 + 8);
    }
    else
    {
      LODWORD(a5) = 0;
    }
    return (unsigned int)a5;
  }
  v12 = *(unsigned int *)(*(_QWORD *)(a1 + 32) + 8LL);
  if ( (int)v12 < 0 )
    v13 = *(_QWORD *)(*(_QWORD *)(a2 + 56) + 16 * (v12 & 0x7FFFFFFF) + 8);
  else
    v13 = *(_QWORD *)(*(_QWORD *)(a2 + 304) + 8 * v12);
  if ( !v13 )
    goto LABEL_19;
  while ( (*(_BYTE *)(v13 + 3) & 0x10) != 0 )
  {
    v13 = *(_QWORD *)(v13 + 32);
    if ( !v13 )
      goto LABEL_19;
  }
  v14 = *(_QWORD *)(v13 + 16);
LABEL_15:
  v15 = sub_36D0D50(v14, a2, a3, a4);
  a5 = v15;
  if ( (_BYTE)v15 )
  {
    v16 = *(_QWORD *)(v13 + 16);
    while ( 1 )
    {
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        break;
      if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 )
      {
        v14 = *(_QWORD *)(v13 + 16);
        if ( v16 != v14 )
          goto LABEL_15;
      }
    }
LABEL_19:
    v17 = *(unsigned int *)(a3 + 8);
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v17 + 1, 8u, a5, a6);
      v17 = *(unsigned int *)(a3 + 8);
    }
    LODWORD(a5) = 1;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v17) = a1;
    ++*(_DWORD *)(a3 + 8);
  }
  return (unsigned int)a5;
}
