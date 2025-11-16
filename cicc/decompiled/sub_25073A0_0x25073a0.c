// Function: sub_25073A0
// Address: 0x25073a0
//
__int64 __fastcall sub_25073A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rdi
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned int v15; // r13d
  unsigned __int64 v17[2]; // [rsp+8h] [rbp-38h] BYREF
  _BYTE v18[40]; // [rsp+18h] [rbp-28h] BYREF

  v7 = *(_BYTE **)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v17[0] = (unsigned __int64)v18;
  v17[1] = 0;
  if ( v8 )
  {
    sub_2506900((__int64)v17, (char **)(a2 + 8), a3, a4, a5, a6);
    if ( !v7 )
      goto LABEL_11;
  }
  else if ( !v7 )
  {
    return 0;
  }
  if ( *v7 != 34 )
  {
    v9 = sub_B46B10((__int64)v7, 0);
    v12 = *a1;
    v13 = v9;
    v14 = *(unsigned int *)(v12 + 8);
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
    {
      sub_C8D5F0(v12, (const void *)(v12 + 16), v14 + 1, 8u, v10, v11);
      v14 = *(unsigned int *)(v12 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v12 + 8 * v14) = v13;
    v15 = 1;
    ++*(_DWORD *)(v12 + 8);
    goto LABEL_7;
  }
LABEL_11:
  v15 = 0;
LABEL_7:
  if ( (_BYTE *)v17[0] != v18 )
    _libc_free(v17[0]);
  return v15;
}
