// Function: sub_15F5720
// Address: 0x15f5720
//
__int64 __fastcall sub_15F5720(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  _QWORD *v5; // rax
  __int64 *v6; // rcx
  __int64 v7; // rdx
  _QWORD *v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned __int64 v11; // rsi
  __int64 v12; // rsi
  int v13; // edx
  int v14; // ecx
  __int64 result; // rax

  sub_15F1EA0(a1, *(_QWORD *)a2, 64, 0, *(_DWORD *)(a2 + 20) & 0xFFFFFFF, 0);
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  *(_DWORD *)(a1 + 56) = v4;
  sub_1648880(a1, v4, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v5 = *(_QWORD **)(a1 - 8);
  else
    v5 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 **)(a2 - 8);
  else
    v6 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v7 )
  {
    v8 = &v5[3 * v7];
    do
    {
      v9 = *v6;
      if ( *v5 )
      {
        v10 = v5[1];
        v11 = v5[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v11 = v10;
        if ( v10 )
          *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
      }
      *v5 = v9;
      if ( v9 )
      {
        v12 = *(_QWORD *)(v9 + 8);
        v5[1] = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = (unsigned __int64)(v5 + 1) | *(_QWORD *)(v12 + 16) & 3LL;
        v5[2] = (v9 + 8) | v5[2] & 3LL;
        *(_QWORD *)(v9 + 8) = v5;
      }
      v5 += 3;
      v6 += 3;
    }
    while ( v8 != v5 );
  }
  v13 = *(_WORD *)(a1 + 18) & 0x8000;
  v14 = *(_WORD *)(a1 + 18) & 0x7FFE;
  result = v13 | v14 | *(unsigned __int16 *)(a2 + 18) & 1u;
  *(_WORD *)(a1 + 18) = v13 | v14 | *(_WORD *)(a2 + 18) & 1;
  return result;
}
