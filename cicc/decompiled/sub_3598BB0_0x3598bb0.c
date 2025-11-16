// Function: sub_3598BB0
// Address: 0x3598bb0
//
unsigned __int64 __fastcall sub_3598BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v9; // r14
  char *v10; // rcx
  __int64 v11; // r14
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rdx
  _QWORD *v15; // rdx
  __int64 v16; // rsi
  int v17; // ecx
  unsigned __int64 result; // rax
  int v19; // esi
  _QWORD *v20; // rcx
  int v21; // esi

  *(_QWORD *)a1 = a3;
  v9 = *(_QWORD *)(a4 + 8) - *(_QWORD *)a4;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v9 )
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3);
    v10 = (char *)sub_22077B0(v9);
  }
  else
  {
    v10 = 0;
  }
  *(_QWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 24) = &v10[v9];
  *(_QWORD *)(a1 + 16) = v10;
  v11 = *(_QWORD *)(a4 + 8) - *(_QWORD *)a4;
  if ( *(_QWORD *)(a4 + 8) != *(_QWORD *)a4 )
    v10 = (char *)memmove(v10, *(const void **)a4, *(_QWORD *)(a4 + 8) - *(_QWORD *)a4);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 16) = &v10[v11];
  v12 = *(_QWORD *)(a5 + 8);
  v13 = *(_DWORD *)(a5 + 24);
  ++*(_QWORD *)a5;
  *(_QWORD *)(a1 + 40) = v12;
  v14 = *(_QWORD *)(a5 + 16);
  *(_QWORD *)(a5 + 8) = 0;
  *(_QWORD *)(a5 + 16) = 0;
  *(_DWORD *)(a5 + 24) = 0;
  *(_QWORD *)(a1 + 48) = v14;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  v15 = *(_QWORD **)(a6 + 8);
  v16 = *(_QWORD *)(a6 + 16);
  v17 = *(_DWORD *)(a6 + 16);
  *(_DWORD *)(a1 + 56) = v13;
  result = *(unsigned int *)(a6 + 24);
  ++*(_QWORD *)a6;
  *(_QWORD *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = v15;
  *(_QWORD *)(a1 + 80) = v16;
  *(_DWORD *)(a1 + 88) = result;
  *(_QWORD *)(a6 + 8) = 0;
  *(_QWORD *)(a6 + 16) = 0;
  *(_DWORD *)(a6 + 24) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  if ( !v17 )
    goto LABEL_7;
  v20 = &v15[2 * (unsigned int)result];
  if ( v15 == v20 )
    goto LABEL_7;
  while ( 1 )
  {
    result = (unsigned __int64)v15;
    if ( *v15 != -8192 && *v15 != -4096 )
      break;
    v15 += 2;
    if ( v20 == v15 )
      goto LABEL_7;
  }
  if ( v15 == v20 )
  {
LABEL_7:
    v19 = 1;
  }
  else
  {
    v21 = 0;
    do
    {
      if ( v21 < *(_DWORD *)(result + 8) )
        v21 = *(_DWORD *)(result + 8);
      result += 16LL;
      *(_DWORD *)(a1 + 96) = v21;
      if ( (_QWORD *)result == v20 )
        break;
      while ( *(_QWORD *)result == -8192 || *(_QWORD *)result == -4096 )
      {
        result += 16LL;
        if ( v20 == (_QWORD *)result )
          goto LABEL_23;
      }
    }
    while ( v20 != (_QWORD *)result );
LABEL_23:
    v19 = v21 + 1;
  }
  *(_DWORD *)(a1 + 96) = v19;
  return result;
}
