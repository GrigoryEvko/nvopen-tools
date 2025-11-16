// Function: sub_103BF70
// Address: 0x103bf70
//
__int64 __fastcall sub_103BF70(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  __int64 v5; // r10
  __int64 v6; // r8
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r11
  __int64 v11; // rdi
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rsi
  bool v19; // zf
  int v21; // edx
  int v22; // ebx

  v4 = *(unsigned int *)(a1 + 88);
  v5 = *(_QWORD *)(a1 + 72);
  v6 = a3;
  if ( (_DWORD)v4 )
  {
    v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( a2 != *v9 )
    {
      v21 = 1;
      while ( v10 != -4096 )
      {
        v22 = v21 + 1;
        v8 = (v4 - 1) & (v21 + v8);
        v9 = (__int64 *)(v5 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v21 = v22;
      }
      return v6;
    }
LABEL_3:
    if ( v9 == (__int64 *)(v5 + 16 * v4) )
      return v6;
    v11 = v9[1];
    v12 = *(_QWORD *)(v11 + 8);
    if ( v11 == v12 )
      return v6;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v12 )
          BUG();
        v13 = *(unsigned __int8 *)(v12 - 32);
        v14 = v12 - 32;
        if ( v13 != 26 )
          break;
        v15 = *(_QWORD *)(v12 - 64);
        if ( !v15 || a4 )
        {
          v16 = v12 - 64;
          goto LABEL_14;
        }
LABEL_8:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v11 == v12 )
          return v6;
      }
      if ( v13 != 27 || (v15 = *(_QWORD *)(v12 - 96)) != 0 && !a4 )
      {
        v6 = v12 - 32;
        goto LABEL_8;
      }
      v16 = v12 - 96;
LABEL_14:
      if ( v15 )
      {
        v17 = *(_QWORD *)(v16 + 8);
        **(_QWORD **)(v16 + 16) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(_QWORD *)(v16 + 16);
      }
      *(_QWORD *)v16 = v6;
      if ( v6 )
      {
        v18 = *(_QWORD *)(v6 + 16);
        *(_QWORD *)(v16 + 8) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = v16 + 8;
        *(_QWORD *)(v16 + 16) = v6 + 16;
        *(_QWORD *)(v6 + 16) = v16;
      }
      v19 = *(_BYTE *)(v12 - 32) == 27;
      v12 = *(_QWORD *)(v12 + 8);
      if ( v19 )
        v6 = v14;
      if ( v11 == v12 )
        return v6;
    }
  }
  return a3;
}
