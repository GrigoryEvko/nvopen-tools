// Function: sub_2AAE740
// Address: 0x2aae740
//
__int64 __fastcall sub_2AAE740(unsigned int **a1, _BYTE *a2)
{
  unsigned int *v3; // r8
  char v4; // r9
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r10
  int v11; // ecx
  int v12; // r13d
  __int64 i; // rcx
  __int64 v14; // rdi
  unsigned int v15; // ecx

  v3 = a1[1];
  v4 = *((_BYTE *)*a1 + 4);
  v5 = **a1;
  if ( v4 )
  {
    if ( !v5 )
      goto LABEL_3;
    v9 = v3[46];
    v10 = *((_QWORD *)v3 + 21);
    if ( !(_DWORD)v9 )
      goto LABEL_3;
    v11 = 37 * v5 - 1;
  }
  else
  {
    if ( v5 <= 1 )
      goto LABEL_3;
    v9 = v3[46];
    v10 = *((_QWORD *)v3 + 21);
    if ( !(_DWORD)v9 )
      goto LABEL_3;
    v11 = 37 * v5;
  }
  v12 = 1;
  for ( i = ((_DWORD)v9 - 1) & (unsigned int)v11; ; i = ((_DWORD)v9 - 1) & v15 )
  {
    v14 = v10 + 72LL * (unsigned int)i;
    if ( v5 == *(_DWORD *)v14 && v4 == *(_BYTE *)(v14 + 4) )
      break;
    if ( *(_DWORD *)v14 == -1 && *(_BYTE *)(v14 + 4) )
      goto LABEL_3;
    v15 = v12 + i;
    ++v12;
  }
  if ( v14 != v10 + 72 * v9 )
  {
    result = sub_B19060(v14 + 8, (__int64)a2, v9, i);
    if ( !(_BYTE)result )
      return result;
    v3 = a1[1];
  }
LABEL_3:
  result = sub_31AC3B0(*((_QWORD *)v3 + 55), a2, *(_QWORD *)a1[2]);
  if ( (_BYTE)result )
  {
    if ( *a2 != 61 )
      return sub_D48480(*((_QWORD *)a1[1] + 52), *((_QWORD *)a2 - 8), v7, v8);
  }
  return result;
}
