// Function: sub_1DABB00
// Address: 0x1dabb00
//
__int64 __fastcall sub_1DABB00(__int64 a1, unsigned int a2, __int64 a3, unsigned int a4, int a5)
{
  unsigned int v6; // r10d
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // rax
  __int64 v10; // r11
  __int64 v11; // r8
  _QWORD *v12; // rdx
  __int64 v13; // r14
  unsigned int v14; // esi
  __int64 v15; // rax
  __int64 v16; // r8
  _QWORD *v17; // rdx
  unsigned int v19; // eax
  int v20; // r10d
  __int64 v21; // rax
  unsigned int v22; // ebx
  __int64 v23; // r8
  _QWORD *v24; // rdx
  unsigned int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // r8
  _QWORD *v28; // rdx

  if ( a5 <= 0 )
  {
    v19 = 9 - a4;
    if ( 9 - a4 > a2 )
      v19 = a2;
    v20 = -a5;
    if ( v19 <= -a5 )
      v20 = v19;
    v21 = 0;
    v22 = v20 + a4;
    if ( v20 )
    {
      do
      {
        v23 = a4++;
        v24 = (_QWORD *)(a3 + 16 * v23);
        *v24 = *(_QWORD *)(a1 + 4 * v21);
        v24[1] = *(_QWORD *)(a1 + 4 * v21 + 8);
        LODWORD(v24) = *(_DWORD *)(a1 + v21 + 144);
        v21 += 4;
        *(_DWORD *)(a3 + 4 * v23 + 144) = (_DWORD)v24;
      }
      while ( a4 != v22 );
    }
    if ( v20 != a2 )
    {
      v25 = v20;
      v26 = 0;
      do
      {
        v27 = v25++;
        v28 = (_QWORD *)(a1 + 16 * v27);
        *(_QWORD *)(a1 + 4 * v26) = *v28;
        *(_QWORD *)(a1 + 4 * v26 + 8) = v28[1];
        *(_DWORD *)(a1 + v26 + 144) = *(_DWORD *)(a1 + 4 * v27 + 144);
        v26 += 4;
      }
      while ( a2 != v25 );
    }
    return (unsigned int)-v20;
  }
  else
  {
    v6 = 9 - a2;
    v7 = a2 - 1;
    if ( 9 - a2 > a4 )
      v6 = a4;
    if ( v6 > a5 )
      v6 = a5;
    if ( a2 )
    {
      v8 = v6 + v7;
      v9 = a1 + 4 * v7 + 144;
      v10 = -3 * a1;
      do
      {
        v11 = v8--;
        v12 = (_QWORD *)(a1 + 16 * v11);
        *v12 = *(_QWORD *)(v10 + 4 * v9 - 576);
        v13 = *(_QWORD *)(v10 + 4 * v9 - 568);
        v9 -= 4;
        v12[1] = v13;
        *(_DWORD *)(a1 + 4 * v11 + 144) = *(_DWORD *)(v9 + 4);
      }
      while ( v9 != a1 + 140 );
    }
    v14 = a4 - v6;
    if ( a4 - v6 != a4 )
    {
      v15 = 0;
      do
      {
        v16 = v14++;
        v17 = (_QWORD *)(a3 + 16 * v16);
        *(_QWORD *)(a1 + 4 * v15) = *v17;
        *(_QWORD *)(a1 + 4 * v15 + 8) = v17[1];
        *(_DWORD *)(a1 + v15 + 144) = *(_DWORD *)(a3 + 4 * v16 + 144);
        v15 += 4;
      }
      while ( v14 != a4 );
    }
    return v6;
  }
}
