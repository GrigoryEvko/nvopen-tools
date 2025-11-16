// Function: sub_15CCFD0
// Address: 0x15ccfd0
//
__int64 __fastcall sub_15CCFD0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  char v7; // si
  __int64 v8; // rcx
  int v9; // edx
  __int64 v10; // rax

  v5 = sub_1648700(a3);
  v6 = *(_QWORD *)(v5 + 40);
  if ( *(_BYTE *)(v5 + 16) != 77 )
    return sub_15CCD40(a1, a2, v6);
  v7 = *(_BYTE *)(v5 + 23) & 0x40;
  v8 = 24LL * *(unsigned int *)(v5 + 56) + 8;
  if ( a2[1] != v6 )
  {
    if ( v7 )
    {
      v10 = *(_QWORD *)(v5 - 8);
      goto LABEL_6;
    }
    v9 = *(_DWORD *)(v5 + 20);
LABEL_5:
    v10 = v5 - 24LL * (v9 & 0xFFFFFFF);
LABEL_6:
    v6 = *(_QWORD *)(v10 + 0xFFFFFFFD55555558LL * (unsigned int)((a3 - v10) >> 3) + v8);
    return sub_15CCD40(a1, a2, v6);
  }
  if ( v7 )
  {
    v10 = *(_QWORD *)(v5 - 8);
    if ( *a2 != *(_QWORD *)(v10 + 0xFFFFFFFD55555558LL * (unsigned int)((a3 - v10) >> 3) + v8) )
      goto LABEL_6;
  }
  else
  {
    v9 = *(_DWORD *)(v5 + 20);
    if ( *a2 != *(_QWORD *)(v5
                          - 24LL * (v9 & 0xFFFFFFF)
                          + 0xFFFFFFFD55555558LL * (unsigned int)((a3 - (v5 - 24LL * (v9 & 0xFFFFFFF))) >> 3)
                          + v8) )
      goto LABEL_5;
  }
  return 1;
}
