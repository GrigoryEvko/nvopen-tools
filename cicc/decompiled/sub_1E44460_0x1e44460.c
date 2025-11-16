// Function: sub_1E44460
// Address: 0x1e44460
//
void __fastcall sub_1E44460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 i; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // r12d
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _BYTE *v15; // r9

  for ( i = *(_QWORD *)(a1 + 32); a1 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( **(_WORD **)(i + 16) && **(_WORD **)(i + 16) != 45 )
      break;
    v8 = *(unsigned int *)(i + 40);
    if ( (_DWORD)v8 != 1 )
    {
      v9 = *(_QWORD *)(i + 32);
      v10 = 1;
      while ( 1 )
      {
        v11 = v10 + 1;
        if ( a2 == *(_QWORD *)(v9 + 40 * v11 + 24) )
          break;
        v10 += 2;
        if ( v10 == (_DWORD)v8 )
          goto LABEL_10;
      }
      sub_1E16C90(i, v11, v8, v9, a5, a6);
      sub_1E16C90(i, v10, v12, v13, v14, v15);
    }
LABEL_10:
    if ( (*(_BYTE *)i & 4) == 0 )
    {
      while ( (*(_BYTE *)(i + 46) & 8) != 0 )
        i = *(_QWORD *)(i + 8);
    }
  }
}
