// Function: sub_31A4290
// Address: 0x31a4290
//
__int64 __fastcall sub_31A4290(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  unsigned __int64 v4; // rax
  _BYTE *v5; // rdi
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r14

  if ( a1 == a2 )
    return 1;
  v2 = sub_D48760(a1, 0);
  if ( !v2 )
    return 0;
  v3 = sub_D47930(a1);
  v4 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == v3 + 48 )
    goto LABEL_22;
  if ( !v4 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_22:
    BUG();
  if ( *(_BYTE *)(v4 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v4 - 20) & 0x7FFFFFF) == 1 )
    return 0;
  v5 = *(_BYTE **)(v4 - 120);
  if ( (unsigned __int8)(*v5 - 82) > 1u )
    return 0;
  v6 = *(_QWORD *)(v2 - 8);
  v7 = 0x1FFFFFFFE0LL;
  v8 = *(_DWORD *)(v2 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(v2 + 4) & 0x7FFFFFF) != 0 )
  {
    v9 = 0;
    do
    {
      if ( v3 == *(_QWORD *)(v6 + 32LL * *(unsigned int *)(v2 + 72) + 8 * v9) )
      {
        v7 = 32 * v9;
        goto LABEL_14;
      }
      ++v9;
    }
    while ( (_DWORD)v8 != (_DWORD)v9 );
    v7 = 0x1FFFFFFFE0LL;
  }
LABEL_14:
  v10 = *((_QWORD *)v5 - 8);
  v11 = *(_QWORD *)(v6 + v7);
  v12 = *((_QWORD *)v5 - 4);
  if ( v10 != v11 || !(unsigned __int8)sub_D48480(a2, *((_QWORD *)v5 - 4), v3, v8) )
  {
    if ( v12 == v11 )
      return sub_D48480(a2, v10, v3, v8);
    return 0;
  }
  return 1;
}
