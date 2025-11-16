// Function: sub_9B4BB0
// Address: 0x9b4bb0
//
__int64 __fastcall sub_9B4BB0(__int64 **a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  int v7; // edx
  unsigned __int64 v8; // rax
  const __m128i *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // r13
  _BYTE *v13; // r8
  __int64 v14; // r15
  char *v15; // r14
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax

  v2 = **a1;
  if ( v2 == *a2 )
    return 1;
  v5 = *(_QWORD *)(*(_QWORD *)(v2 - 8)
                 + 32LL * *(unsigned int *)(v2 + 72)
                 + 8LL * (unsigned int)(((__int64)a2 - *(_QWORD *)(v2 - 8)) >> 5));
  v6 = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == v5 + 48 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    v7 = *(unsigned __int8 *)(v6 - 24);
    v8 = v6 - 24;
    if ( (unsigned int)(v7 - 30) >= 0xB )
      v8 = 0;
  }
  a1[1][5] = v8;
  v9 = (const __m128i *)a1[1];
  v10 = *a2;
  v11 = v9[2].m128i_i64[1];
  if ( *(_BYTE *)v11 != 31 )
    return sub_9A6530(v10, (__int64)a1[2], v9, *(_DWORD *)a1[3]);
  if ( (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != 3 )
    return sub_9A6530(v10, (__int64)a1[2], v9, *(_DWORD *)a1[3]);
  v13 = *(_BYTE **)(v11 - 96);
  if ( *v13 != 82 )
    return sub_9A6530(v10, (__int64)a1[2], v9, *(_DWORD *)a1[3]);
  v14 = *((_QWORD *)v13 - 8);
  v15 = (char *)*((_QWORD *)v13 - 4);
  if ( v10 == v14 && v15 )
  {
    v16 = (unsigned int)sub_B53900(*(_QWORD *)(v11 - 96));
  }
  else
  {
    if ( (char *)v10 != v15 || !v14 )
      return sub_9A6530(v10, (__int64)a1[2], v9, *(_DWORD *)a1[3]);
    v15 = (char *)*((_QWORD *)v13 - 8);
    v16 = (unsigned int)sub_B53960(*(_QWORD *)(v11 - 96));
  }
  v17 = *(_QWORD *)(v11 - 32);
  if ( !v17 )
    goto LABEL_20;
  v18 = *(_QWORD *)(v11 - 64);
  if ( !v18 )
    goto LABEL_20;
  v19 = *(_QWORD *)(**a1 + 40);
  if ( (v19 == v17) == (v19 == v18) )
    goto LABEL_20;
  if ( v19 == v18 )
    LODWORD(v16) = sub_B52870(v16);
  if ( !(unsigned __int8)sub_9867F0(v16, v15) )
  {
LABEL_20:
    v9 = (const __m128i *)a1[1];
    v10 = *a2;
    return sub_9A6530(v10, (__int64)a1[2], v9, *(_DWORD *)a1[3]);
  }
  return 1;
}
