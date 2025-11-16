// Function: sub_EE1480
// Address: 0xee1480
//
void __fastcall sub_EE1480(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // r13
  unsigned __int64 v5; // rbx
  __int64 i; // r15
  __int64 v7; // r13
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // r8
  bool v10; // cc
  unsigned int v11; // eax

  v3 = a2 - a1;
  v5 = a2;
  if ( (__int64)(a2 - a1) > 16 )
  {
    for ( i = ((v3 >> 4) - 2) / 2; ; --i )
    {
      sub_ED7AF0(a1, i, v3 >> 4, *(_QWORD *)(a1 + 16 * i), *(_QWORD *)(a1 + 16 * i + 8));
      if ( !i )
        break;
    }
  }
  v7 = v3 >> 4;
  if ( a2 < a3 )
  {
    while ( 1 )
    {
      v10 = *(_DWORD *)v5 <= *(_DWORD *)a1;
      if ( *(_DWORD *)v5 < *(_DWORD *)a1 )
        goto LABEL_7;
      if ( *(_DWORD *)v5 == *(_DWORD *)a1 )
      {
        v11 = *(_DWORD *)(a1 + 4);
        v10 = *(_DWORD *)(v5 + 4) <= v11;
        if ( *(_DWORD *)(v5 + 4) < v11 )
          goto LABEL_7;
      }
      if ( !v10 )
        goto LABEL_8;
      if ( *(_QWORD *)(v5 + 8) < *(_QWORD *)(a1 + 8) )
      {
LABEL_7:
        v8 = *(_QWORD *)v5;
        v9 = *(_QWORD *)(v5 + 8);
        *(_QWORD *)v5 = *(_QWORD *)a1;
        *(_QWORD *)(v5 + 8) = *(_QWORD *)(a1 + 8);
        sub_ED7AF0(a1, 0, v7, v8, v9);
LABEL_8:
        v5 += 16LL;
        if ( a3 <= v5 )
          return;
      }
      else
      {
        v5 += 16LL;
        if ( a3 <= v5 )
          return;
      }
    }
  }
}
