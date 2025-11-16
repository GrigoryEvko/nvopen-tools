// Function: sub_F1FA60
// Address: 0xf1fa60
//
bool __fastcall sub_F1FA60(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v8; // rcx
  int v9; // esi
  _QWORD *v10; // rbx
  int v11; // r8d
  unsigned int i; // eax
  _QWORD *v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rsi

  if ( !*(_BYTE *)(a1 + 1128) )
    sub_F1F440(a1, a2, a3, a4, a5, a6);
  if ( (*(_BYTE *)(a1 + 992) & 1) != 0 )
  {
    v8 = a1 + 1000;
    v9 = 7;
    v10 = (_QWORD *)(a1 + 1128);
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 1000);
    v15 = *(unsigned int *)(a1 + 1008);
    v10 = (_QWORD *)(v8 + 16 * v15);
    if ( !(_DWORD)v15 )
      return 0;
    v9 = v15 - 1;
  }
  v11 = 1;
  for ( i = v9
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v9 & v14 )
  {
    v13 = (_QWORD *)(v8 + 16LL * i);
    if ( a2 == *v13 && a3 == (__int64 *)v13[1] )
      break;
    if ( *v13 == -4096 && v13[1] == -4096 )
      return 0;
    v14 = v11 + i;
    ++v11;
  }
  return v13 != v10;
}
