// Function: sub_C657C0
// Address: 0xc657c0
//
__int64 __fastcall sub_C657C0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v8; // rcx
  int v9; // r12d
  _QWORD v10[2]; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE v11[176]; // [rsp+20h] [rbp-B0h] BYREF

  v5 = *((_DWORD *)a1 + 3) + 1;
  if ( v5 > 2 * *((_DWORD *)a1 + 2) )
  {
    sub_C65A40(a1, a4);
    v8 = *a1;
    v9 = *((_DWORD *)a1 + 2);
    v10[0] = v11;
    v10[1] = 0x2000000000LL;
    a3 = (__int64 *)(v8
                   + 8LL
                   * ((*(unsigned int (__fastcall **)(__int64 *, __int64 *, _QWORD *))(a4 + 16))(a1, a2, v10) & (v9 - 1)));
    if ( (_BYTE *)v10[0] != v11 )
      _libc_free(v10[0], a2);
    v5 = *((_DWORD *)a1 + 3) + 1;
  }
  *((_DWORD *)a1 + 3) = v5;
  result = *a3;
  if ( !*a3 )
    result = (unsigned __int64)a3 | 1;
  *a2 = result;
  *a3 = (__int64)a2;
  return result;
}
