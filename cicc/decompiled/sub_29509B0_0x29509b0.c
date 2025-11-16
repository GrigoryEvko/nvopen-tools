// Function: sub_29509B0
// Address: 0x29509b0
//
__int64 __fastcall sub_29509B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 i; // r15
  unsigned __int8 *v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax

  v3 = *(_QWORD *)(a1 + 80);
  for ( i = v3 + 8LL * *(unsigned int *)(a1 + 88); v3 != i; i -= 8 )
  {
    while ( 1 )
    {
      v5 = *(unsigned __int8 **)(i - 8);
      if ( *(_BYTE *)a2 > 0x15u )
        break;
      a2 = sub_96F480((unsigned int)*v5 - 29, a2, *((_QWORD *)v5 + 1), *(_QWORD *)(a1 + 240));
      if ( !a2 )
        break;
      i -= 8;
      if ( v3 == i )
        return a2;
    }
    v6 = sub_B47F80(v5);
    v7 = (_QWORD *)v6;
    if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
      v8 = *(_QWORD *)(v6 - 8);
    else
      v8 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)v8 )
    {
      v9 = *(_QWORD *)(v8 + 8);
      **(_QWORD **)(v8 + 16) = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
    }
    *(_QWORD *)v8 = a2;
    if ( a2 )
    {
      v10 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(v8 + 8) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = v8 + 8;
      *(_QWORD *)(v8 + 16) = a2 + 16;
      *(_QWORD *)(a2 + 16) = v8;
    }
    v11 = *(_QWORD *)(a1 + 224);
    if ( !v11 )
      BUG();
    a2 = (__int64)v7;
    sub_B44150(v7, *(_QWORD *)(v11 + 16), *(unsigned __int64 **)(a1 + 224), *(_QWORD *)(a1 + 232));
  }
  return a2;
}
