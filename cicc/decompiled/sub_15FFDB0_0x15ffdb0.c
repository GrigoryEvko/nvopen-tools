// Function: sub_15FFDB0
// Address: 0x15ffdb0
//
__int64 __fastcall sub_15FFDB0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  unsigned int v4; // ecx
  __int64 v5; // r9
  __int64 v6; // rsi
  __int64 *v7; // r8
  __int64 *v8; // rdi
  __int64 v9; // r9
  unsigned __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int64 v12; // rcx
  __int64 v13; // r11
  _QWORD *v14; // rcx
  __int64 v15; // r12
  unsigned __int64 v16; // rbx
  __int64 v17; // rbx
  _QWORD *v18; // rcx
  __int64 v19; // r9
  __int64 v20; // r11
  unsigned __int64 v21; // r10
  __int64 v22; // r10

  result = a1;
  v4 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 24LL * v4;
  v6 = v4 - 2;
  v7 = (__int64 *)(v5 + 24 * v6);
  v8 = (__int64 *)(v5 + 24LL * (v4 - 1));
  if ( 2 * a3 + 4 != v4 )
  {
    v13 = *v7;
    v14 = (_QWORD *)(v5 + 24LL * (unsigned int)(2 * a3 + 2));
    if ( *v14 )
    {
      v15 = v14[1];
      v16 = v14[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v16 = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
    }
    *v14 = v13;
    if ( v13 )
    {
      v17 = *(_QWORD *)(v13 + 8);
      v14[1] = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = (unsigned __int64)(v14 + 1) | *(_QWORD *)(v17 + 16) & 3LL;
      v14[2] = (v13 + 8) | v14[2] & 3LL;
      *(_QWORD *)(v13 + 8) = v14;
    }
    v18 = (_QWORD *)(v5 + 24LL * (unsigned int)(2 * a3 + 3));
    v19 = *v8;
    if ( *v18 )
    {
      v20 = v18[1];
      v21 = v18[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v21 = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
    }
    *v18 = v19;
    if ( v19 )
    {
      v22 = *(_QWORD *)(v19 + 8);
      v18[1] = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = (unsigned __int64)(v18 + 1) | *(_QWORD *)(v22 + 16) & 3LL;
      v18[2] = (v19 + 8) | v18[2] & 3LL;
      *(_QWORD *)(v19 + 8) = v18;
    }
  }
  if ( *v7 )
  {
    v9 = v7[1];
    v10 = v7[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  *v7 = 0;
  if ( *v8 )
  {
    v11 = v8[1];
    v12 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v12 = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
  }
  *v8 = 0;
  *(_DWORD *)(result + 20) = v6 & 0xFFFFFFF | *(_DWORD *)(result + 20) & 0xF0000000;
  return result;
}
