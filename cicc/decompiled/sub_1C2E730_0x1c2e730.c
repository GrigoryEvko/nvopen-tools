// Function: sub_1C2E730
// Address: 0x1c2e730
//
__int64 __fastcall sub_1C2E730(__int64 a1, const void *a2, size_t a3, _QWORD *a4)
{
  unsigned int v5; // eax
  unsigned int v6; // r12d
  __int64 v8; // r14
  _QWORD *v9; // rax
  unsigned __int64 v10[2]; // [rsp+0h] [rbp-B0h] BYREF
  _BYTE v11[160]; // [rsp+10h] [rbp-A0h] BYREF

  v10[0] = (unsigned __int64)v11;
  v10[1] = 0x1000000000LL;
  LOBYTE(v5) = sub_1C2E100(a1, a2, a3, (__int64)v10, 1);
  v6 = v5;
  if ( (_BYTE)v5 )
  {
    v8 = *(_QWORD *)v10[0];
    if ( (_BYTE *)v10[0] != v11 )
      _libc_free(v10[0]);
    v9 = *(_QWORD **)(v8 + 24);
    if ( *(_DWORD *)(v8 + 32) > 0x40u )
      v9 = (_QWORD *)*v9;
    *a4 = v9;
  }
  else if ( (_BYTE *)v10[0] != v11 )
  {
    _libc_free(v10[0]);
  }
  return v6;
}
