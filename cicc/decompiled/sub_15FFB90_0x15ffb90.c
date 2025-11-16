// Function: sub_15FFB90
// Address: 0x15ffb90
//
__int64 __fastcall sub_15FFB90(__int64 a1, __int64 a2)
{
  unsigned int v4; // ecx
  __int64 *v5; // rax
  unsigned int v6; // r8d
  unsigned int v7; // eax
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned int i; // edx
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r10
  unsigned __int64 v14; // r9
  __int64 v15; // r9
  _QWORD *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r10
  unsigned __int64 v19; // r9
  __int64 v20; // r9
  __int64 result; // rax

  sub_15F1EA0(a1, *(_QWORD *)a2, 3, 0, 0, 0);
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 24LL * v4);
  sub_15FF940(a1, *v5, v5[3], v4);
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = v6 | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_DWORD *)(a1 + 20) = v7;
  if ( (v7 & 0x40000000) != 0 )
    v8 = *(_QWORD *)(a1 - 8);
  else
    v8 = a1 - 24LL * v6;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v9 = *(_QWORD *)(a2 - 8);
  else
    v9 = a2 - 24LL * v6;
  if ( v6 != 2 )
  {
    for ( i = 2; i != v6; i += 2 )
    {
      v11 = (_QWORD *)(v8 + 24LL * i);
      v12 = *(_QWORD *)(v9 + 24LL * i);
      if ( *v11 )
      {
        v13 = v11[1];
        v14 = v11[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v14 = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
      }
      *v11 = v12;
      if ( v12 )
      {
        v15 = *(_QWORD *)(v12 + 8);
        v11[1] = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = (unsigned __int64)(v11 + 1) | *(_QWORD *)(v15 + 16) & 3LL;
        v11[2] = (v12 + 8) | v11[2] & 3LL;
        *(_QWORD *)(v12 + 8) = v11;
      }
      v16 = (_QWORD *)(v8 + 24LL * (i + 1));
      v17 = *(_QWORD *)(v9 + 24LL * (i + 1));
      if ( *v16 )
      {
        v18 = v16[1];
        v19 = v16[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v19 = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
      }
      *v16 = v17;
      if ( v17 )
      {
        v20 = *(_QWORD *)(v17 + 8);
        v16[1] = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = (unsigned __int64)(v16 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
        v16[2] = (v17 + 8) | v16[2] & 3LL;
        *(_QWORD *)(v17 + 8) = v16;
      }
    }
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
