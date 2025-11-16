// Function: sub_1DAB460
// Address: 0x1dab460
//
void __fastcall sub_1DAB460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  int v7; // r9d
  unsigned int i; // eax
  __int64 v9; // r8
  __int64 v10; // rdi
  _QWORD *v11; // rsi
  _QWORD *v12; // rcx
  int v13; // r9d

  v5 = *(_QWORD *)a1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 80LL) )
  {
    sub_1DAB250(a1, 1, v5, a4, a5);
  }
  else
  {
    v7 = *(_DWORD *)(v5 + 84);
    for ( i = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4) + 1;
          v7 != i;
          *(_DWORD *)(v5 + 4 * v10 + 64) = *(_DWORD *)(v5 + 4 * v9 + 64) )
    {
      v9 = i;
      v10 = i++ - 1;
      v11 = (_QWORD *)(v5 + 16 * v9);
      v12 = (_QWORD *)(v5 + 16 * v10);
      *v12 = *v11;
      v12[1] = v11[1];
    }
    v13 = v7 - 1;
    *(_DWORD *)(v5 + 84) = v13;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v13;
  }
}
