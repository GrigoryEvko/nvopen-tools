// Function: sub_15F80C0
// Address: 0x15f80c0
//
__int64 __fastcall sub_15F80C0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  __int64 v5; // r9
  __int64 *v6; // rcx
  _QWORD *v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rdi
  unsigned __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 result; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r12

  sub_15F1EA0(
    a1,
    *(_QWORD *)a2,
    *(unsigned __int8 *)(a2 + 16) - 24,
    a1 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
    *(_DWORD *)(a2 + 20) & 0xFFFFFFF,
    0);
  v4 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v5 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v6 = (__int64 *)(a2 - v5);
  if ( v5 )
  {
    v7 = &v4[(unsigned __int64)v5 / 8];
    do
    {
      v8 = *v6;
      if ( *v4 )
      {
        v9 = v4[1];
        v10 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v10 = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
      }
      *v4 = v8;
      if ( v8 )
      {
        v11 = *(_QWORD *)(v8 + 8);
        v4[1] = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = (unsigned __int64)(v4 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
        v4[2] = (v8 + 8) | v4[2] & 3LL;
        *(_QWORD *)(v8 + 8) = v4;
      }
      v4 += 3;
      v6 += 3;
    }
    while ( v4 != v7 );
  }
  result = *(_QWORD *)(a2 - 24);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v13 = *(_QWORD *)(a1 - 16);
    v14 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v14 = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
  }
  *(_QWORD *)(a1 - 24) = result;
  if ( result )
  {
    v15 = *(_QWORD *)(result + 8);
    *(_QWORD *)(a1 - 16) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = (a1 - 16) | *(_QWORD *)(v15 + 16) & 3LL;
    v16 = *(_QWORD *)(a1 - 8);
    v17 = a1 - 24;
    *(_QWORD *)(v17 + 16) = (result + 8) | v16 & 3;
    *(_QWORD *)(result + 8) = v17;
  }
  return result;
}
