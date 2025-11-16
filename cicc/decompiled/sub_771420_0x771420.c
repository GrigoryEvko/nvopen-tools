// Function: sub_771420
// Address: 0x771420
//
__int64 __fastcall sub_771420(__int64 a1, __int64 a2)
{
  unsigned int v4; // edi
  int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  _DWORD *i; // rax
  _QWORD *v9; // rax

  v4 = *(_DWORD *)(a2 + 32);
  v5 = *(_DWORD *)(a1 + 64);
  v6 = *(_QWORD *)(a1 + 56);
  v7 = v5 & v4;
  for ( i = (_DWORD *)(v6 + 4LL * (v5 & v4)); v4 != *i; i = (_DWORD *)(v6 + 4LL * v7) )
    v7 = v5 & (v7 + 1);
  *i = 0;
  if ( *(_DWORD *)(v6 + 4LL * ((v7 + 1) & v5)) )
  {
    sub_771390(*(_QWORD *)(a1 + 56), *(_DWORD *)(a1 + 64), v7);
    --*(_DWORD *)(a1 + 68);
    v9 = *(_QWORD **)(a2 + 8);
    if ( v9 )
      goto LABEL_5;
  }
  else
  {
    --*(_DWORD *)(a1 + 68);
    v9 = *(_QWORD **)(a2 + 8);
    if ( v9 )
    {
LABEL_5:
      *v9 = *(_QWORD *)a2;
      goto LABEL_6;
    }
  }
  *(_QWORD *)(a1 + 184) = *(_QWORD *)a2;
LABEL_6:
  if ( *(_QWORD *)a2 )
    *(_QWORD *)(*(_QWORD *)a2 + 8LL) = *(_QWORD *)(a2 + 8);
  return j___libc_free(a2, *(unsigned int *)(a2 + 36));
}
