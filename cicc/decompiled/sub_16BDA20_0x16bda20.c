// Function: sub_16BDA20
// Address: 0x16bda20
//
__int64 __fastcall sub_16BDA20(__int64 *a1, __int64 *a2, __int64 *a3)
{
  unsigned int v4; // eax
  __int64 result; // rax
  int v6; // r12d
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned __int64 v9[2]; // [rsp+0h] [rbp-C0h] BYREF
  _BYTE v10[176]; // [rsp+10h] [rbp-B0h] BYREF

  v4 = *((_DWORD *)a1 + 5) + 1;
  if ( v4 > 2 * *((_DWORD *)a1 + 4) )
  {
    sub_16BDC90();
    v6 = *((_DWORD *)a1 + 4);
    v7 = a1[1];
    v9[1] = 0x2000000000LL;
    v8 = *a1;
    v9[0] = (unsigned __int64)v10;
    a3 = (__int64 *)(v7
                   + 8LL
                   * ((*(unsigned int (__fastcall **)(__int64 *, __int64 *, unsigned __int64 *))(v8 + 24))(a1, a2, v9)
                    & (v6 - 1)));
    if ( (_BYTE *)v9[0] != v10 )
      _libc_free(v9[0]);
    v4 = *((_DWORD *)a1 + 5) + 1;
  }
  *((_DWORD *)a1 + 5) = v4;
  result = *a3;
  if ( !*a3 )
    result = (unsigned __int64)a3 | 1;
  *a2 = result;
  *a3 = (__int64)a2;
  return result;
}
