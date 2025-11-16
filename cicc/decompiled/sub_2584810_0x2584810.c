// Function: sub_2584810
// Address: 0x2584810
//
__int64 __fastcall sub_2584810(signed int **a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rax
  unsigned int v7; // r12d
  char *v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v10[2]; // [rsp+8h] [rbp-28h] BYREF
  _BYTE v11[24]; // [rsp+18h] [rbp-18h] BYREF

  v6 = *a2;
  v10[0] = (unsigned __int64)v11;
  v10[1] = 0;
  v9 = v6;
  if ( *((_DWORD *)a2 + 4) )
    sub_2538240((__int64)v10, a2 + 1, a3, a4, a5, a6);
  v7 = sub_2584510(a1, (__int64)&v9, a3, a4, a5, a6);
  if ( (_BYTE *)v10[0] != v11 )
    _libc_free(v10[0]);
  return v7;
}
