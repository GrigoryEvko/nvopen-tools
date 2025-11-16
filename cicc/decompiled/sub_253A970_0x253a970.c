// Function: sub_253A970
// Address: 0x253a970
//
__int64 __fastcall sub_253A970(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v11[2]; // [rsp+8h] [rbp-38h] BYREF
  _BYTE v12[40]; // [rsp+18h] [rbp-28h] BYREF

  v6 = *a2;
  v7 = *((_DWORD *)a2 + 4);
  v11[0] = (unsigned __int64)v12;
  v11[1] = 0;
  if ( v7 )
    sub_2538240((__int64)v11, (char **)a2 + 1, a3, a4, a5, a6);
  v8 = *a1;
  v9 = *(unsigned int *)(*a1 + 8);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12) )
  {
    sub_C8D5F0(v8, (const void *)(v8 + 16), v9 + 1, 8u, a5, a6);
    v9 = *(unsigned int *)(v8 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v8 + 8 * v9) = v6;
  ++*(_DWORD *)(v8 + 8);
  if ( (_BYTE *)v11[0] != v12 )
    _libc_free(v11[0]);
  return 1;
}
