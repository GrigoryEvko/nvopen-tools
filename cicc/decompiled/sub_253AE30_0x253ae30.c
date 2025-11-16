// Function: sub_253AE30
// Address: 0x253ae30
//
__int64 __fastcall sub_253AE30(__int64 ***a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  int v8; // eax
  __int64 *v9; // r13
  _BYTE *v10; // r12
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned int v13; // r12d
  unsigned __int64 v15[2]; // [rsp+18h] [rbp-38h] BYREF
  _BYTE v16[40]; // [rsp+28h] [rbp-28h] BYREF

  v7 = *(_QWORD *)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v15[0] = (unsigned __int64)v16;
  v15[1] = 0;
  if ( v8 )
    sub_2538240((__int64)v15, (char **)(a2 + 8), a3, a4, a5, a6);
  v9 = **a1;
  v10 = *(_BYTE **)(v7 - 32);
  if ( v10 && *v10 )
    v10 = 0;
  v11 = sub_B491C0(v7);
  LOBYTE(v12) = sub_DFE030(v9, v11, (__int64)v10);
  v13 = v12;
  if ( (_BYTE *)v15[0] != v16 )
    _libc_free(v15[0]);
  return v13;
}
