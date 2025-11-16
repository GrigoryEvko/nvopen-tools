// Function: sub_14C8500
// Address: 0x14c8500
//
__int64 __fastcall sub_14C8500(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // r8
  __int64 v4; // rdx
  __int64 v5; // r13
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rax

  v1 = sub_1648700(a1);
  if ( *(_BYTE *)(v1 + 16) != 78 )
    return 0;
  v4 = *(_QWORD *)(v1 - 24);
  v5 = 0;
  if ( !*(_BYTE *)(v4 + 16) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
    v5 = v1;
  if ( *(_BYTE *)(v4 + 16)
    || *(_DWORD *)(v4 + 36) != 4
    || *a1 == *(_QWORD *)((v1 & 0xFFFFFFFFFFFFFFF8LL)
                        - 24LL * (*(_DWORD *)((v1 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)) )
  {
    return 0;
  }
  v6 = sub_1648720(a1);
  if ( *(char *)(v5 + 23) >= 0
    || ((v7 = sub_1648A40(v5), v9 = v7 + v8, *(char *)(v5 + 23) >= 0) ? (v10 = 0) : (v10 = sub_1648A40(v5)), v10 == v9) )
  {
LABEL_20:
    BUG();
  }
  while ( 1 )
  {
    v2 = v10;
    if ( v6 >= *(_DWORD *)(v10 + 8) && v6 < *(_DWORD *)(v10 + 12) )
      return v2;
    v10 += 16;
    if ( v9 == v10 )
      goto LABEL_20;
  }
}
