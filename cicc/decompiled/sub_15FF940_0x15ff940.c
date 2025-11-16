// Function: sub_15FF940
// Address: 0x15ff940
//
__int64 __fastcall sub_15FF940(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  int v5; // eax
  __int64 result; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rbx

  v5 = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 56) = a4;
  *(_DWORD *)(a1 + 20) = v5 & 0xF0000000 | 2;
  sub_1648880(a1, a4, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    result = *(_QWORD *)(a1 - 8);
  else
    result = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( *(_QWORD *)result )
  {
    v7 = *(_QWORD *)(result + 8);
    v8 = *(_QWORD *)(result + 16) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v8 = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
  }
  *(_QWORD *)result = a2;
  if ( a2 )
  {
    v9 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(result + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = (result + 8) | *(_QWORD *)(v9 + 16) & 3LL;
    *(_QWORD *)(result + 16) = (a2 + 8) | *(_QWORD *)(result + 16) & 3LL;
    *(_QWORD *)(a2 + 8) = result;
  }
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v10 = *(_QWORD **)(a1 - 8);
  }
  else
  {
    result = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v10 = (_QWORD *)(a1 - result);
  }
  if ( v10[3] )
  {
    v11 = v10[4];
    result = v10[5] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v11;
    if ( v11 )
    {
      result |= *(_QWORD *)(v11 + 16) & 3LL;
      *(_QWORD *)(v11 + 16) = result;
    }
  }
  v10[3] = a3;
  if ( a3 )
  {
    v12 = *(_QWORD *)(a3 + 8);
    v10[4] = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (unsigned __int64)(v10 + 4) | *(_QWORD *)(v12 + 16) & 3LL;
    v13 = v10[5];
    v14 = v10 + 3;
    result = (a3 + 8) | v13 & 3;
    v14[2] = result;
    *(_QWORD *)(a3 + 8) = v14;
  }
  return result;
}
