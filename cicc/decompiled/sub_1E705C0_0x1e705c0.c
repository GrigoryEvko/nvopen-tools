// Function: sub_1E705C0
// Address: 0x1e705c0
//
void __fastcall sub_1E705C0(__int64 a1, unsigned __int64 *a2, __int64 *a3)
{
  __int64 v5; // rax
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  _QWORD *v10; // rdi

  v5 = *(_QWORD *)(a1 + 928);
  if ( a2 != (unsigned __int64 *)v5 )
  {
    if ( a2 == (unsigned __int64 *)a3 )
      goto LABEL_9;
    if ( !a2 )
      BUG();
    goto LABEL_4;
  }
  if ( !a2 )
    BUG();
  if ( (*(_BYTE *)a2 & 4) == 0 && (*((_BYTE *)a2 + 46) & 8) != 0 )
  {
    do
      v5 = *(_QWORD *)(v5 + 8);
    while ( (*(_BYTE *)(v5 + 46) & 8) != 0 );
  }
  *(_QWORD *)(a1 + 928) = *(_QWORD *)(v5 + 8);
  if ( a2 != (unsigned __int64 *)a3 )
  {
LABEL_4:
    v6 = a2;
    if ( (*a2 & 4) == 0 && (*((_BYTE *)a2 + 46) & 8) != 0 )
    {
      do
        v6 = (unsigned __int64 *)v6[1];
      while ( (*((_BYTE *)v6 + 46) & 8) != 0 );
    }
    v7 = (unsigned __int64 *)v6[1];
    if ( a2 != v7 && a3 != (__int64 *)v7 && v7 != a2 )
    {
      v8 = *v7;
      *(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v7;
      v8 &= 0xFFFFFFFFFFFFFFF8LL;
      *v7 = *v7 & 7 | *a2 & 0xFFFFFFFFFFFFFFF8LL;
      v9 = *a3;
      *(_QWORD *)(v8 + 8) = a3;
      v9 &= 0xFFFFFFFFFFFFFFF8LL;
      *a2 = v9 | *a2 & 7;
      *(_QWORD *)(v9 + 8) = a2;
      *a3 = v8 | *a3 & 7;
    }
  }
LABEL_9:
  v10 = *(_QWORD **)(a1 + 2112);
  if ( v10 )
    sub_1DC1A70(v10, (unsigned __int64)a2, 1);
  if ( *(__int64 **)(a1 + 928) == a3 )
    *(_QWORD *)(a1 + 928) = a2;
}
