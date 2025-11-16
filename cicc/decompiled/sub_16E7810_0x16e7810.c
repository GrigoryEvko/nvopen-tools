// Function: sub_16E7810
// Address: 0x16e7810
//
__int64 __fastcall sub_16E7810(__int64 a1, char *a2, unsigned __int64 a3)
{
  unsigned __int64 v5; // rbx
  size_t v6; // rdx
  __int64 v7; // rdi
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r15d

  v5 = a3;
  *(_QWORD *)(a1 + 64) += a3;
  while ( 1 )
  {
    v6 = 0x40000000;
    v7 = *(unsigned int *)(a1 + 36);
    if ( v5 <= 0x40000000 )
      v6 = v5;
    result = write(v7, a2, v6);
    if ( result < 0 )
      break;
    a2 += result;
    v5 -= result;
LABEL_6:
    if ( !v5 )
      return result;
  }
  result = (__int64)__errno_location();
  v12 = *(_DWORD *)result;
  if ( *(_DWORD *)result == 4 || v12 == 11 )
    goto LABEL_6;
  result = sub_2241E50(v7, a2, v9, v10, v11);
  *(_DWORD *)(a1 + 48) = v12;
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
