// Function: sub_15F7BC0
// Address: 0x15f7bc0
//
__int64 __fastcall sub_15F7BC0(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // ecx
  int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 *v7; // rax
  int v8; // r10d
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // r9
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // r8
  unsigned __int64 v16; // rdi
  __int64 v17; // rdi

  sub_15F1EA0(a1, *(_QWORD *)a2, 10, 0, *(_DWORD *)(a2 + 20) & 0xFFFFFFF, 0);
  v2 = *(_BYTE *)(a2 + 23);
  v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    v4 = v2 & 0x40;
    if ( v4 )
      v5 = *(_QWORD *)(a2 - 8);
    else
      v5 = a2 - 24LL * v3;
    v6 = *(_QWORD *)(v5 + 24);
  }
  else
  {
    LOBYTE(v4) = v2 & 0x40;
    v6 = 0;
  }
  if ( (_BYTE)v4 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 24LL * v3);
  sub_15F79C0(a1, *v7, v6, v3);
  v8 = *(_DWORD *)(a1 + 56);
  v9 = v8 & 0xFFFFFFF;
  result = (unsigned int)v9 | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_DWORD *)(a1 + 20) = result;
  if ( (result & 0x40000000) != 0 )
  {
    v11 = *(_QWORD *)(a1 - 8);
  }
  else
  {
    result = 24 * v9;
    v11 = a1 - 24 * v9;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v12 = *(_QWORD *)(a2 - 8);
  }
  else
  {
    result = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v12 = a2 - result;
  }
  v13 = 1;
  if ( v8 != 1 )
  {
    do
    {
      result = v11 + 24LL * v13;
      v14 = *(_QWORD *)(v12 + 24LL * v13);
      if ( *(_QWORD *)result )
      {
        v15 = *(_QWORD *)(result + 8);
        v16 = *(_QWORD *)(result + 16) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v16 = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
      }
      *(_QWORD *)result = v14;
      if ( v14 )
      {
        v17 = *(_QWORD *)(v14 + 8);
        *(_QWORD *)(result + 8) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = (result + 8) | *(_QWORD *)(v17 + 16) & 3LL;
        *(_QWORD *)(result + 16) = (v14 + 8) | *(_QWORD *)(result + 16) & 3LL;
        *(_QWORD *)(v14 + 8) = result;
      }
      ++v13;
    }
    while ( v8 != v13 );
  }
  return result;
}
