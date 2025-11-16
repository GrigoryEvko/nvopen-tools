// Function: sub_15F79C0
// Address: 0x15f79c0
//
_QWORD *__fastcall sub_15F79C0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  _QWORD *result; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  _QWORD *v9; // rbx
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx

  *(_DWORD *)(a1 + 56) = a4;
  *(_DWORD *)(a1 + 20) = (1 - ((a3 == 0) - 1)) | *(_DWORD *)(a1 + 20) & 0xF0000000;
  sub_1648880(a1, a4, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    result = *(_QWORD **)(a1 - 8);
  else
    result = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( *result )
  {
    v6 = result[1];
    v7 = result[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v7 = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
  }
  *result = a2;
  if ( a2 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    result[1] = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(v8 + 16) & 3LL;
    result[2] = (a2 + 8) | result[2] & 3LL;
    *(_QWORD *)(a2 + 8) = result;
  }
  if ( a3 )
  {
    *(_WORD *)(a1 + 18) = *(_WORD *)(a1 + 18) & 0x8000 | *(_WORD *)(a1 + 18) & 0x7FFE | 1;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v9 = *(_QWORD **)(a1 - 8);
    else
      v9 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v9[3] )
    {
      v10 = v9[4];
      v11 = v9[5] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    v9[3] = a3;
    v12 = *(_QWORD *)(a3 + 8);
    v9[4] = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (unsigned __int64)(v9 + 4) | *(_QWORD *)(v12 + 16) & 3LL;
    result = (_QWORD *)(v9[5] & 3LL | (a3 + 8));
    v9[5] = result;
    *(_QWORD *)(a3 + 8) = v9 + 3;
  }
  return result;
}
