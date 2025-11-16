// Function: sub_ADDCE0
// Address: 0xaddce0
//
__int64 __fastcall sub_ADDCE0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  int v4; // eax
  __int64 *v5; // rdi
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // r13
  __int64 *v14; // rdi
  __int64 v15; // rdi
  int v16; // r12d
  __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 112);
  if ( *(_DWORD *)(a1 + 116) <= (unsigned int)v3 )
  {
    v7 = a1 + 120;
    v8 = a1 + 104;
    v13 = (__int64 *)sub_C8D7D0(a1 + 104, a1 + 120, 0, 8, v17);
    v14 = &v13[*(unsigned int *)(a1 + 112)];
    if ( v14 )
    {
      *v14 = a2;
      if ( a2 )
        sub_B96E90(v14, a2, 1);
    }
    result = sub_ADDB20(v8, v13, v9, v10, v11, v12);
    v15 = *(_QWORD *)(a1 + 104);
    v16 = v17[0];
    if ( v7 != v15 )
      result = _libc_free(v15, v13);
    *(_QWORD *)(a1 + 104) = v13;
    *(_DWORD *)(a1 + 116) = v16;
    ++*(_DWORD *)(a1 + 112);
  }
  else
  {
    v4 = *(_DWORD *)(a1 + 112);
    v5 = (__int64 *)(*(_QWORD *)(a1 + 104) + 8 * v3);
    if ( v5 )
    {
      *v5 = a2;
      if ( a2 )
        sub_B96E90(v5, a2, 1);
      v4 = *(_DWORD *)(a1 + 112);
    }
    result = (unsigned int)(v4 + 1);
    *(_DWORD *)(a1 + 112) = result;
  }
  return result;
}
