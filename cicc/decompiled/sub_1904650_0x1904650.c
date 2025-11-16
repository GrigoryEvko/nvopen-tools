// Function: sub_1904650
// Address: 0x1904650
//
void __fastcall sub_1904650(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 i; // r12
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rdi
  unsigned int v14; // r8d
  __int64 *v15; // rcx
  __int64 *v16; // rax

  v4 = a2 + 72;
  v5 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 == v5 )
  {
    i = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v5 + 24);
      if ( i != v5 + 16 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v4 == v5 )
        return;
      if ( !v5 )
        BUG();
    }
  }
  while ( v4 != v5 )
  {
    if ( !i )
      BUG();
    v7 = i - 24;
    if ( *(_BYTE *)(*(_QWORD *)(i - 24) + 8LL) != 16 )
    {
      v8 = *(unsigned __int8 *)(i - 8) - 24;
      if ( v8 > 0x28 )
      {
        if ( *(_BYTE *)(i - 8) != 76 )
          goto LABEL_14;
        v10 = *(unsigned __int16 *)(i - 6);
        BYTE1(v10) &= ~0x80u;
        v11 = (unsigned int)(v10 - 1);
        if ( (unsigned int)v11 > 0xD || *(_DWORD *)&asc_42BDC60[4 * v11] == 42 )
          goto LABEL_14;
        v12 = *(__int64 **)(a3 + 8);
        if ( *(__int64 **)(a3 + 16) != v12 )
          goto LABEL_36;
        v13 = &v12[*(unsigned int *)(a3 + 28)];
        v14 = *(_DWORD *)(a3 + 28);
        if ( v12 != v13 )
        {
          v15 = 0;
          while ( v7 != *v12 )
          {
            if ( *v12 == -2 )
              v15 = v12;
            if ( v13 == ++v12 )
            {
LABEL_31:
              if ( !v15 )
                goto LABEL_44;
              *v15 = v7;
              --*(_DWORD *)(a3 + 32);
              ++*(_QWORD *)a3;
              goto LABEL_14;
            }
          }
          goto LABEL_14;
        }
      }
      else
      {
        if ( v8 <= 0x26 )
          goto LABEL_14;
        v16 = *(__int64 **)(a3 + 8);
        if ( *(__int64 **)(a3 + 16) != v16 )
        {
LABEL_36:
          sub_16CCBA0(a3, v7);
          goto LABEL_14;
        }
        v13 = &v16[*(unsigned int *)(a3 + 28)];
        v14 = *(_DWORD *)(a3 + 28);
        if ( v16 != v13 )
        {
          v15 = 0;
          while ( v7 != *v16 )
          {
            if ( *v16 == -2 )
              v15 = v16;
            if ( v13 == ++v16 )
              goto LABEL_31;
          }
          goto LABEL_14;
        }
      }
LABEL_44:
      if ( v14 >= *(_DWORD *)(a3 + 24) )
        goto LABEL_36;
      *(_DWORD *)(a3 + 28) = v14 + 1;
      *v13 = v7;
      ++*(_QWORD *)a3;
    }
LABEL_14:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v5 + 24) )
    {
      v9 = v5 - 24;
      if ( !v5 )
        v9 = 0;
      if ( i != v9 + 40 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v4 == v5 )
        return;
      if ( !v5 )
        BUG();
    }
  }
}
