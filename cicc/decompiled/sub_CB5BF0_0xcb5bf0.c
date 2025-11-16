// Function: sub_CB5BF0
// Address: 0xcb5bf0
//
__int64 __fastcall sub_CB5BF0(__int64 a1, char *a2, unsigned __int64 a3)
{
  __int64 *v6; // rdi
  size_t v7; // rdx
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // r15d

  v6 = *(__int64 **)(a1 + 64);
  if ( v6 && v6[4] != v6[2] )
    sub_CB5AE0(v6);
  *(_QWORD *)(a1 + 88) += a3;
  while ( 1 )
  {
    v7 = 0x40000000;
    v8 = *(unsigned int *)(a1 + 48);
    if ( a3 <= 0x40000000 )
      v7 = a3;
    result = write(v8, a2, v7);
    if ( result < 0 )
      break;
    a2 += result;
    a3 -= result;
LABEL_9:
    if ( !a3 )
      return result;
  }
  result = (__int64)__errno_location();
  v13 = *(_DWORD *)result;
  if ( *(_DWORD *)result == 4 || v13 == 11 )
    goto LABEL_9;
  result = sub_2241E50(v8, a2, v10, v11, v12);
  *(_DWORD *)(a1 + 72) = v13;
  *(_QWORD *)(a1 + 80) = result;
  return result;
}
