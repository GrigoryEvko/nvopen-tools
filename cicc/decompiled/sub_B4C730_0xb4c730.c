// Function: sub_B4C730
// Address: 0xb4c730
//
__int64 __fastcall sub_B4C730(__int64 a1, unsigned __int8 *a2, unsigned int a3)
{
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rax
  unsigned __int8 *v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx

  sub_B44260(a1, *((_QWORD *)a2 + 1), *a2 - 29, a3, 0, 0);
  v5 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v6 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  v7 = a1 - v5;
  v8 = &a2[-v6];
  if ( v6 )
  {
    v9 = v7 + v6;
    do
    {
      v10 = *(_QWORD *)v8;
      if ( *(_QWORD *)v7 )
      {
        v11 = *(_QWORD *)(v7 + 8);
        **(_QWORD **)(v7 + 16) = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = *(_QWORD *)(v7 + 16);
      }
      *(_QWORD *)v7 = v10;
      if ( v10 )
      {
        v12 = *(_QWORD *)(v10 + 16);
        *(_QWORD *)(v7 + 8) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = v7 + 8;
        *(_QWORD *)(v7 + 16) = v10 + 16;
        *(_QWORD *)(v10 + 16) = v7;
      }
      v7 += 32;
      v8 += 32;
    }
    while ( v7 != v9 );
  }
  result = *((_QWORD *)a2 - 4);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v14 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = result;
  if ( result )
  {
    v15 = *(_QWORD *)(result + 16);
    *(_QWORD *)(a1 - 24) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = result + 16;
    *(_QWORD *)(result + 16) = a1 - 32;
  }
  return result;
}
