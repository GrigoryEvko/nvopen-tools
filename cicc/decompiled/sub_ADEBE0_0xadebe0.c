// Function: sub_ADEBE0
// Address: 0xadebe0
//
__int64 __fastcall sub_ADEBE0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        __int64 a10,
        int a11,
        __int64 a12,
        __int64 a13,
        char a14,
        __int64 a15)
{
  __int64 v15; // r10
  int v19; // ebx
  __int64 v20; // rdi
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  int v25; // eax
  __int64 *v26; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 *v30; // rbx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 *v33; // rdi
  __int64 v34; // rdi
  int v35; // r15d
  __int64 v38[7]; // [rsp+28h] [rbp-38h] BYREF

  v15 = a3;
  v19 = (int)a2;
  if ( a2 && *a2 == 17 )
    v19 = 0;
  v20 = *(_QWORD *)(a1 + 8);
  v38[0] = a15;
  if ( a13 )
  {
    sub_B9B140(v20, a12, a13);
    v15 = a3;
  }
  v21 = 0;
  if ( a4 )
    v21 = sub_B9B140(v20, v15, a4);
  v22 = sub_B065E0(v20, 4, v21, a5, a6, v19, a10, a7, a8, 0, (unsigned __int8)(a14 != 0) << 24, a9, a11);
  v23 = *(unsigned int *)(a1 + 64);
  v24 = v22;
  v25 = v23;
  if ( *(_DWORD *)(a1 + 68) <= (unsigned int)v23 )
  {
    v30 = (__int64 *)sub_C8D7D0(a1 + 56, a1 + 72, 0, 8, v38);
    v33 = &v30[*(unsigned int *)(a1 + 64)];
    if ( v33 )
    {
      *v33 = v24;
      if ( v24 )
        sub_B96E90(v33, v24, 1);
    }
    sub_ADDB20(a1 + 56, v30, v28, v29, v31, v32);
    v34 = *(_QWORD *)(a1 + 56);
    v35 = v38[0];
    if ( a1 + 72 != v34 )
      _libc_free(v34, v30);
    ++*(_DWORD *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = v30;
    *(_DWORD *)(a1 + 68) = v35;
  }
  else
  {
    v26 = (__int64 *)(*(_QWORD *)(a1 + 56) + 8 * v23);
    if ( v26 )
    {
      *v26 = v24;
      if ( v24 )
        sub_B96E90(v26, v24, 1);
      v25 = *(_DWORD *)(a1 + 64);
    }
    *(_DWORD *)(a1 + 64) = v25 + 1;
  }
  sub_ADDDC0(a1, v24);
  return v24;
}
