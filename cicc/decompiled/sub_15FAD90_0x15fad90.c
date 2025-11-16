// Function: sub_15FAD90
// Address: 0x15fad90
//
__int64 __fastcall sub_15FAD90(__int64 a1, __int64 a2, __int64 a3, const void *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  size_t v15; // r15
  unsigned __int64 v16; // rbx

  if ( *(_QWORD *)(a1 - 48) )
  {
    v8 = *(_QWORD *)(a1 - 40);
    v9 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v9 = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 + 16) & 3LL | v9;
  }
  *(_QWORD *)(a1 - 48) = a2;
  if ( a2 )
  {
    v10 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 40) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = (a1 - 40) | *(_QWORD *)(v10 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (a2 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(a2 + 8) = a1 - 48;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v11 = *(_QWORD *)(a1 - 16);
    v12 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v13 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (a1 - 16) | *(_QWORD *)(v13 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a3 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a3 + 8) = a1 - 24;
  }
  v14 = *(unsigned int *)(a1 + 64);
  v15 = 4 * a5;
  v16 = (4 * a5) >> 2;
  if ( v16 > (unsigned __int64)*(unsigned int *)(a1 + 68) - v14 )
  {
    sub_16CD150(a1 + 56, a1 + 72, v16 + v14, 4);
    v14 = *(unsigned int *)(a1 + 64);
  }
  if ( v15 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 56) + 4 * v14), a4, v15);
    LODWORD(v14) = *(_DWORD *)(a1 + 64);
  }
  *(_DWORD *)(a1 + 64) = v16 + v14;
  return sub_164B780(a1, a6);
}
