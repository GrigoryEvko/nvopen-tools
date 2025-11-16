// Function: sub_1DC9E80
// Address: 0x1dc9e80
//
__int64 __fastcall sub_1DC9E80(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r13
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rdi
  _QWORD *v9; // rbx
  __int64 v10; // rdx
  unsigned __int64 *v11; // r13
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rdi

  v2 = *(unsigned __int64 **)(a1 + 304);
  v3 = &v2[2 * *(unsigned int *)(a1 + 312)];
  while ( v2 != v3 )
  {
    v4 = *v2;
    v2 += 2;
    _libc_free(v4);
  }
  v5 = *(unsigned int *)(a1 + 264);
  *(_DWORD *)(a1 + 312) = 0;
  if ( (_DWORD)v5 )
  {
    v9 = *(_QWORD **)(a1 + 256);
    *(_QWORD *)(a1 + 320) = 0;
    v10 = *v9;
    v11 = &v9[v5];
    v12 = v9 + 1;
    *(_QWORD *)(a1 + 240) = v10;
    *(_QWORD *)(a1 + 248) = v10 + 4096;
    while ( v11 != v12 )
    {
      v13 = *v12++;
      _libc_free(v13);
    }
    *(_DWORD *)(a1 + 264) = 1;
  }
  sub_1DC9D90(a1 + 344);
  v6 = *(_QWORD *)(a1 + 416);
  while ( v6 )
  {
    sub_1DC9780(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7, 48);
  }
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = a1 + 408;
  *(_QWORD *)(a1 + 432) = a1 + 408;
  *(_QWORD *)(a1 + 440) = 0;
  return a1 + 408;
}
