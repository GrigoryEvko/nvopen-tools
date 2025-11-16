// Function: sub_F18070
// Address: 0xf18070
//
__int64 __fastcall sub_F18070(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  int v8; // eax
  _QWORD *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // r14
  unsigned __int64 *v13; // r13
  unsigned __int64 *v14; // rdi
  unsigned __int64 v15; // rax
  int v16; // r12d
  unsigned __int64 v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v7 )
  {
    v12 = a1 + 16;
    v13 = (unsigned __int64 *)sub_C8D7D0(a1, a1 + 16, 0, 0x18u, v17, a6);
    v14 = &v13[3 * *(unsigned int *)(a1 + 8)];
    if ( v14 )
    {
      v15 = *a2;
      *v14 = 6;
      v14[1] = 0;
      v14[2] = v15;
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD73F0((__int64)v14);
    }
    result = sub_F17F80(a1, v13);
    v16 = v17[0];
    if ( v12 != *(_QWORD *)a1 )
      result = _libc_free(*(_QWORD *)a1, v13);
    *(_QWORD *)a1 = v13;
    *(_DWORD *)(a1 + 12) = v16;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 8);
    v9 = (_QWORD *)(*(_QWORD *)a1 + 24 * v7);
    if ( v9 )
    {
      v10 = *a2;
      *v9 = 6;
      v9[1] = 0;
      v9[2] = v10;
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD73F0((__int64)v9);
      v8 = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)(v8 + 1);
    *(_DWORD *)(a1 + 8) = result;
  }
  return result;
}
