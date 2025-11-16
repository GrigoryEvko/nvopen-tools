// Function: sub_398E9E0
// Address: 0x398e9e0
//
__int64 __fastcall sub_398E9E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v9; // rax
  unsigned __int64 v10; // r12
  _QWORD *v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned __int64 *v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v17; // rdi

  sub_398DFB0(a1, a2, a4, a5, *(_QWORD *)(a3 + 8));
  v9 = (_QWORD *)sub_22077B0(0x48u);
  v10 = (unsigned __int64)v9;
  if ( v9 )
  {
    *v9 = a4;
    v11 = v9 + 7;
    *(v11 - 6) = a5;
    *(v11 - 5) = 0;
    *((_DWORD *)v11 - 8) = -1;
    *(v11 - 3) = 0;
    *(_QWORD *)(v10 + 40) = v11;
    *(_QWORD *)(v10 + 48) = 0x100000000LL;
  }
  v12 = *(_DWORD *)(a1 + 672);
  if ( v12 >= *(_DWORD *)(a1 + 676) )
  {
    sub_398E850(a1 + 664, 0);
    v12 = *(_DWORD *)(a1 + 672);
  }
  v13 = *(_QWORD *)(a1 + 664);
  v14 = (unsigned __int64 *)(v13 + 8LL * v12);
  if ( v14 )
  {
    *v14 = v10;
    v13 = *(_QWORD *)(a1 + 664);
    v15 = (unsigned int)(*(_DWORD *)(a1 + 672) + 1);
    *(_DWORD *)(a1 + 672) = v15;
  }
  else
  {
    v15 = v12 + 1;
    *(_DWORD *)(a1 + 672) = v15;
    if ( v10 )
    {
      v17 = *(_QWORD *)(v10 + 40);
      if ( v17 != v10 + 56 )
        _libc_free(v17);
      j_j___libc_free_0(v10);
      v15 = *(unsigned int *)(a1 + 672);
      v13 = *(_QWORD *)(a1 + 664);
    }
  }
  sub_39A0D10(a1 + 4040, a3, *(_QWORD *)(v13 + 8 * v15 - 8));
  return *(_QWORD *)(*(_QWORD *)(a1 + 664) + 8LL * *(unsigned int *)(a1 + 672) - 8);
}
