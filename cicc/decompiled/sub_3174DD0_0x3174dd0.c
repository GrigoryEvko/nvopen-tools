// Function: sub_3174DD0
// Address: 0x3174dd0
//
__int64 __fastcall sub_3174DD0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 **a5)
{
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r11
  unsigned int *v10; // rcx
  unsigned int v11; // ebx
  __int64 v12; // r12
  unsigned int v13; // r13d

  result = a2 - 1;
  v8 = (a2 - 1) / 2;
  if ( a2 <= a3 )
  {
    *(_DWORD *)(a1 + 4 * a2) = a4;
    return result;
  }
  v9 = 176LL * a4;
  while ( 1 )
  {
    v10 = (unsigned int *)(a1 + 4 * v8);
    v11 = *v10;
    v12 = **a5;
    v13 = *(_DWORD *)(v12 + v9 + 104);
    result = 176LL * *v10;
    if ( *(_DWORD *)(v12 + result + 104) == v13 )
      break;
    if ( *(_DWORD *)(v12 + result + 104) <= v13 )
      goto LABEL_8;
LABEL_5:
    *(_DWORD *)(a1 + 4 * a2) = v11;
    a2 = v8;
    result = (v8 - 1) / 2;
    if ( a3 >= v8 )
      goto LABEL_9;
    v8 = (v8 - 1) / 2;
  }
  if ( v11 > a4 )
    goto LABEL_5;
LABEL_8:
  v10 = (unsigned int *)(a1 + 4 * a2);
LABEL_9:
  *v10 = a4;
  return result;
}
