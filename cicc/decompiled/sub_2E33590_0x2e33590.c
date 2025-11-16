// Function: sub_2E33590
// Address: 0x2e33590
//
__int64 *__fastcall sub_2E33590(__int64 a1, __int64 *a2, char a3)
{
  char *v5; // rax
  char *v6; // rdx
  char *v7; // rsi
  unsigned int *v8; // rsi
  __int64 v9; // rdx
  int v10; // eax

  if ( *(_QWORD *)(a1 + 152) != *(_QWORD *)(a1 + 144) )
  {
    v5 = (char *)sub_2E32F70(a1, (__int64)a2);
    v6 = *(char **)(a1 + 152);
    v7 = v5 + 4;
    if ( v6 != v5 + 4 )
    {
      memmove(v5, v7, v6 - v7);
      v7 = *(char **)(a1 + 152);
    }
    v8 = (unsigned int *)(v7 - 4);
    *(_QWORD *)(a1 + 152) = v8;
    if ( a3 )
      sub_2E33470(*(unsigned int **)(a1 + 144), v8);
  }
  sub_2E32230(*a2, a1);
  v9 = *(_QWORD *)(a1 + 112) + 8LL * *(unsigned int *)(a1 + 120);
  v10 = *(_DWORD *)(a1 + 120);
  if ( (__int64 *)v9 != a2 + 1 )
  {
    memmove(a2, a2 + 1, v9 - (_QWORD)(a2 + 1));
    v10 = *(_DWORD *)(a1 + 120);
  }
  *(_DWORD *)(a1 + 120) = v10 - 1;
  return a2;
}
