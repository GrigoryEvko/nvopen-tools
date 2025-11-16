// Function: sub_2B0A8B0
// Address: 0x2b0a8b0
//
__int64 __fastcall sub_2B0A8B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  __int64 v4; // r10
  __int64 v5; // r11
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  v4 = a1;
  if ( a2 == a3 )
    return a1;
  v5 = a1 + a3 - a2;
  v6 = (a3 - a1) >> 4;
  v7 = (a2 - a1) >> 4;
  if ( v7 == v6 - v7 )
  {
    v20 = a2;
    do
    {
      v21 = *(_QWORD *)(v20 + 8);
      v22 = *(_QWORD *)(v4 + 8);
      v4 += 16;
      v20 += 16;
      *(_QWORD *)(v4 - 8) = v21;
      LODWORD(v21) = *(_DWORD *)(v20 - 12);
      *(_QWORD *)(v20 - 8) = v22;
      LODWORD(v22) = *(_DWORD *)(v4 - 12);
      *(_DWORD *)(v4 - 12) = v21;
      LODWORD(v21) = *(_DWORD *)(v20 - 16);
      *(_DWORD *)(v20 - 12) = v22;
      LODWORD(v22) = *(_DWORD *)(v4 - 16);
      *(_DWORD *)(v4 - 16) = v21;
      *(_DWORD *)(v20 - 16) = v22;
    }
    while ( a2 != v4 );
    return a2;
  }
  v8 = v6 - v7;
  if ( v7 >= v6 - v7 )
    goto LABEL_12;
  while ( 1 )
  {
    v9 = v4 + 16 * v7;
    if ( v8 > 0 )
    {
      v10 = v4;
      v11 = 0;
      do
      {
        v12 = *(_QWORD *)(v10 + 8);
        v13 = *(_QWORD *)(v9 + 8);
        ++v11;
        v10 += 16;
        v9 += 16;
        *(_QWORD *)(v10 - 8) = v13;
        LODWORD(v13) = *(_DWORD *)(v9 - 12);
        *(_QWORD *)(v9 - 8) = v12;
        LODWORD(v12) = *(_DWORD *)(v10 - 12);
        *(_DWORD *)(v10 - 12) = v13;
        LODWORD(v13) = *(_DWORD *)(v9 - 16);
        *(_DWORD *)(v9 - 12) = v12;
        LODWORD(v12) = *(_DWORD *)(v10 - 16);
        *(_DWORD *)(v10 - 16) = v13;
        *(_DWORD *)(v9 - 16) = v12;
      }
      while ( v8 != v11 );
      v4 += 16 * v8;
    }
    if ( !(v6 % v7) )
      break;
    v8 = v7;
    v7 -= v6 % v7;
    while ( 1 )
    {
      v6 = v8;
      v8 -= v7;
      if ( v7 < v8 )
        break;
LABEL_12:
      v14 = v4 + 16 * v6;
      v4 = v14 - 16 * v8;
      if ( v7 > 0 )
      {
        v15 = v14 - 16 * v8;
        v16 = 0;
        do
        {
          v17 = *(_QWORD *)(v15 - 8);
          v18 = *(_QWORD *)(v14 - 8);
          ++v16;
          v15 -= 16;
          v14 -= 16;
          *(_QWORD *)(v15 + 8) = v18;
          LODWORD(v18) = *(_DWORD *)(v14 + 4);
          *(_QWORD *)(v14 + 8) = v17;
          LODWORD(v17) = *(_DWORD *)(v15 + 4);
          *(_DWORD *)(v15 + 4) = v18;
          LODWORD(v18) = *(_DWORD *)v14;
          *(_DWORD *)(v14 + 4) = v17;
          LODWORD(v17) = *(_DWORD *)v15;
          *(_DWORD *)v15 = v18;
          *(_DWORD *)v14 = v17;
        }
        while ( v7 != v16 );
        v4 -= 16 * v7;
      }
      v7 = v6 % v8;
      if ( !(v6 % v8) )
        return v5;
    }
  }
  return v5;
}
