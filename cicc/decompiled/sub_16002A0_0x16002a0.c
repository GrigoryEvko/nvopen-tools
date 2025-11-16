// Function: sub_16002A0
// Address: 0x16002a0
//
__int64 __fastcall sub_16002A0(__int64 a1, __int64 a2)
{
  int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rcx
  _QWORD *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 result; // rax

  v4 = *(_DWORD *)(a2 + 20);
  v5 = sub_16498A0(a2);
  v6 = sub_1643270(v5);
  sub_15F1EA0(a1, v6, 4, 0, v4 & 0xFFFFFFF, 0);
  sub_1648880(a1, *(_DWORD *)(a2 + 20) & 0xFFFFFFF, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v7 = *(_QWORD **)(a1 - 8);
  else
    v7 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v8 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v9 = *(__int64 **)(a2 - 8);
  else
    v9 = (__int64 *)(a2 - 24LL * (unsigned int)v8);
  if ( (_DWORD)v8 )
  {
    v10 = &v7[3 * v8];
    do
    {
      v11 = *v9;
      if ( *v7 )
      {
        v12 = v7[1];
        v13 = v7[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v13 = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
      }
      *v7 = v11;
      if ( v11 )
      {
        v14 = *(_QWORD *)(v11 + 8);
        v7[1] = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = (unsigned __int64)(v7 + 1) | *(_QWORD *)(v14 + 16) & 3LL;
        v7[2] = (v11 + 8) | v7[2] & 3LL;
        *(_QWORD *)(v11 + 8) = v7;
      }
      v7 += 3;
      v9 += 3;
    }
    while ( v7 != v10 );
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
