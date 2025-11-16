// Function: sub_B4FD20
// Address: 0xb4fd20
//
__int64 __fastcall sub_B4FD20(__int64 a1, __int64 a2, __int64 a3, const void *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  size_t v12; // r13
  __int64 v13; // rax
  __int64 v14; // rbx

  if ( *(_QWORD *)(a1 - 64) )
  {
    v8 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a2;
  if ( a2 )
  {
    v9 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 56) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v10 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  v12 = 4 * a5;
  v13 = *(unsigned int *)(a1 + 80);
  v14 = (4 * a5) >> 2;
  if ( v14 + v13 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, a1 + 88, v14 + v13, 4);
    v13 = *(unsigned int *)(a1 + 80);
  }
  if ( v12 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 72) + 4 * v13), a4, v12);
    LODWORD(v13) = *(_DWORD *)(a1 + 80);
  }
  *(_DWORD *)(a1 + 80) = v13 + v14;
  return sub_BD6B50(a1, a6);
}
