// Function: sub_37BBBE0
// Address: 0x37bbbe0
//
unsigned __int64 __fastcall sub_37BBBE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  unsigned int i; // esi
  __int64 v11; // rdx
  int v12; // esi
  __int64 v13; // rax
  int v14; // r13d
  __int64 v15; // rdx
  unsigned __int64 result; // rax
  unsigned __int16 *v17; // r13
  unsigned __int16 *v18; // rbx
  int v19; // r14d
  int v20; // esi

  v7 = *(_QWORD *)(a1 + 408);
  v8 = *(_QWORD *)(v7 + 832);
  v9 = *(unsigned int *)(v7 + 848);
  if ( (_DWORD)v9 )
  {
    a5 = (unsigned int)(v9 - 1);
    a6 = 1;
    for ( i = (v9 - 1) & 0xD1533BD0; ; i = a5 & v12 )
    {
      while ( 1 )
      {
        v11 = v8 + 8LL * i;
        if ( *(_WORD *)v11 != 8 )
          break;
        if ( !*(_WORD *)(v11 + 2) )
          goto LABEL_8;
        v20 = a6 + i;
        a6 = (unsigned int)(a6 + 1);
        i = a5 & v20;
      }
      if ( *(_WORD *)v11 == 0xFFFF && *(_WORD *)(v11 + 2) == 0xFFFF )
        break;
      v12 = a6 + i;
      a6 = (unsigned int)(a6 + 1);
    }
  }
  v11 = v8 + 8 * v9;
LABEL_8:
  v13 = *(unsigned int *)(a2 + 8);
  v14 = *(_DWORD *)(v11 + 4);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 4u, a5, a6);
    v13 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v13) = v14;
  ++*(_DWORD *)(a2 + 8);
  v15 = *(_QWORD *)(a1 + 408);
  result = *(unsigned int *)(v15 + 840);
  if ( !(_DWORD)result )
    return result;
  result = *(_QWORD *)(v15 + 832);
  v17 = (unsigned __int16 *)(result + 8LL * *(unsigned int *)(v15 + 848));
  if ( (unsigned __int16 *)result == v17 )
    return result;
  while ( 1 )
  {
    v18 = (unsigned __int16 *)result;
    if ( *(_WORD *)result != 0xFFFF )
      break;
    if ( *(_WORD *)(result + 2) != 0xFFFF )
      goto LABEL_15;
LABEL_33:
    result += 8LL;
    if ( v17 == (unsigned __int16 *)result )
      return result;
  }
  if ( *(_WORD *)result == 0xFFFE && *(_WORD *)(result + 2) == 0xFFFE )
    goto LABEL_33;
LABEL_15:
  if ( v17 == (unsigned __int16 *)result )
    return result;
LABEL_19:
  if ( v18[1] )
  {
    result = *(unsigned int *)(a2 + 8);
    v19 = *((_DWORD *)v18 + 1);
    if ( result + 1 > *(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 4u, a5, a6);
      result = *(unsigned int *)(a2 + 8);
    }
    *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v19;
    ++*(_DWORD *)(a2 + 8);
  }
  for ( v18 += 4; v17 != v18; v18 += 4 )
  {
    result = *v18;
    if ( (_WORD)result == 0xFFFF )
    {
      if ( v18[1] != 0xFFFF )
        goto LABEL_18;
    }
    else if ( (_WORD)result != 0xFFFE || v18[1] != 0xFFFE )
    {
LABEL_18:
      if ( v17 != v18 )
        goto LABEL_19;
      return result;
    }
  }
  return result;
}
