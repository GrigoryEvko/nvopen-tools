// Function: sub_15E0C30
// Address: 0x15e0c30
//
__int64 __fastcall sub_15E0C30(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v7; // rdi
  _QWORD *v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx

  v2 = a1 + 72;
  v3 = *(_QWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 32) &= ~0x400000u;
  if ( v3 == a1 + 72 )
    goto LABEL_9;
  do
  {
    v4 = v3 - 24;
    if ( !v3 )
      v4 = 0;
    sub_157EE90(v4);
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v2 != v3 );
  while ( v2 != (*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v5 = *(_QWORD *)(a1 + 80);
    if ( v5 )
      v5 -= 24;
    sub_157F980(v5);
LABEL_9:
    ;
  }
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 0 )
    return sub_161FB70(a1);
  v7 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v8 = *(_QWORD **)(a1 - 8);
    v9 = &v8[v7];
  }
  else
  {
    v8 = (_QWORD *)(a1 - v7 * 8);
    v9 = (_QWORD *)a1;
  }
  do
  {
    if ( *v8 )
    {
      v10 = v8[1];
      v11 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    *v8 = 0;
    v8 += 3;
  }
  while ( v9 != v8 );
  *(_DWORD *)(a1 + 20) &= 0xF0000000;
  *(_WORD *)(a1 + 18) &= 0xFFF1u;
  return sub_161FB70(a1);
}
