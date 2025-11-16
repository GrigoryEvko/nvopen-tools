// Function: sub_14214B0
// Address: 0x14214b0
//
__int64 __fastcall sub_14214B0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  __int64 v5; // r8
  unsigned int v7; // ecx
  __int64 v8; // rdi
  __int64 *v9; // rdx
  __int64 v10; // r11
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r11
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  int v17; // ecx
  __int64 v18; // rsi
  int v20; // edx
  int v21; // ebx

  v4 = *(unsigned int *)(a1 + 80);
  v5 = a3;
  if ( (_DWORD)v4 )
  {
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = *(_QWORD *)(a1 + 64);
    v9 = (__int64 *)(v8 + 16LL * v7);
    v10 = *v9;
    if ( a2 != *v9 )
    {
      v20 = 1;
      while ( v10 != -8 )
      {
        v21 = v20 + 1;
        v7 = (v4 - 1) & (v20 + v7);
        v9 = (__int64 *)(v8 + 16LL * v7);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v20 = v21;
      }
      return v5;
    }
LABEL_3:
    if ( v9 == (__int64 *)(v8 + 16 * v4) )
      return v5;
    v11 = v9[1];
    v12 = *(_QWORD *)(v11 + 8);
    if ( v11 == v12 )
      return v5;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v12 )
          BUG();
        v18 = v12 - 32;
        v17 = *(unsigned __int8 *)(v12 - 16);
        if ( (unsigned int)(v17 - 21) <= 1 )
          break;
        v12 = *(_QWORD *)(v12 + 8);
        v5 = v18;
        if ( v11 == v12 )
          return v5;
      }
      if ( !*(_QWORD *)(v12 - 56) )
        break;
      if ( a4 )
      {
        v13 = *(_QWORD *)(v12 - 48);
        v14 = v12 - 56;
        v15 = *(_QWORD *)(v12 - 40) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v15 = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v15;
        goto LABEL_10;
      }
LABEL_15:
      v12 = *(_QWORD *)(v12 + 8);
      if ( (_BYTE)v17 == 22 )
        v5 = v18;
      if ( v11 == v12 )
        return v5;
    }
    v14 = v12 - 56;
LABEL_10:
    *(_QWORD *)(v12 - 56) = v5;
    if ( v5 )
    {
      v16 = *(_QWORD *)(v5 + 8);
      *(_QWORD *)(v12 - 48) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = (v12 - 48) | *(_QWORD *)(v16 + 16) & 3LL;
      *(_QWORD *)(v12 - 40) = (v5 + 8) | *(_QWORD *)(v12 - 40) & 3LL;
      *(_QWORD *)(v5 + 8) = v14;
    }
    LOBYTE(v17) = *(_BYTE *)(v12 - 16);
    goto LABEL_15;
  }
  return a3;
}
