// Function: sub_1600500
// Address: 0x1600500
//
__int64 __fastcall sub_1600500(__int64 a1, int a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 *v5; // rcx
  _QWORD *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // r9
  unsigned __int64 v9; // r8
  __int64 v10; // r8
  __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 24LL * v2;
  v4 = v2 - 1;
  v5 = (__int64 *)(v3 + 24 * v4);
  v6 = (_QWORD *)(v3 + 24LL * (unsigned int)(a2 + 1));
  v7 = *v5;
  if ( *v6 )
  {
    v8 = v6[1];
    v9 = v6[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v9 = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 + 16) & 3LL | v9;
  }
  *v6 = v7;
  if ( v7 )
  {
    v10 = *(_QWORD *)(v7 + 8);
    v6[1] = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = (unsigned __int64)(v6 + 1) | *(_QWORD *)(v10 + 16) & 3LL;
    v6[2] = (v7 + 8) | v6[2] & 3LL;
    *(_QWORD *)(v7 + 8) = v6;
  }
  if ( *v5 )
  {
    v11 = v5[1];
    v12 = v5[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *v5 = 0;
  result = v4 & 0xFFFFFFF | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_DWORD *)(a1 + 20) = result;
  return result;
}
