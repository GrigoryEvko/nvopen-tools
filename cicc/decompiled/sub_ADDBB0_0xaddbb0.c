// Function: sub_ADDBB0
// Address: 0xaddbb0
//
__int64 __fastcall sub_ADDBB0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        char a10,
        int a11,
        int a12,
        __int64 a13)
{
  int v13; // r10d
  __int64 v16; // r12
  __int64 v18; // rdx
  int v19; // eax
  __int64 *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 *v25; // r13
  __int64 *v26; // rdi
  int v27; // r15d
  _QWORD v28[7]; // [rsp+8h] [rbp-38h] BYREF

  v13 = 0;
  if ( a5 )
    v13 = sub_B9B140(a1, a4, a5);
  v16 = sub_B0C150(a1, a3, v13, a7, a8, a9, a6, a11, a12, a13, 0, 1);
  if ( a10 )
  {
    v18 = *(unsigned int *)(a2 + 8);
    v19 = v18;
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v18 )
    {
      v25 = (__int64 *)sub_C8D7D0(a2, a2 + 16, 0, 8, v28);
      v26 = &v25[*(unsigned int *)(a2 + 8)];
      if ( v26 )
      {
        *v26 = v16;
        if ( v16 )
          sub_B96E90(v26, v16, 1);
      }
      sub_ADDB20(a2, v25, v21, v22, v23, v24);
      v27 = v28[0];
      if ( a2 + 16 != *(_QWORD *)a2 )
        _libc_free(*(_QWORD *)a2, v25);
      ++*(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v25;
      *(_DWORD *)(a2 + 12) = v27;
    }
    else
    {
      v20 = (__int64 *)(*(_QWORD *)a2 + 8 * v18);
      if ( v20 )
      {
        *v20 = v16;
        if ( v16 )
          sub_B96E90(v20, v16, 1);
        v19 = *(_DWORD *)(a2 + 8);
      }
      *(_DWORD *)(a2 + 8) = v19 + 1;
    }
  }
  return v16;
}
