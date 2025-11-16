// Function: sub_25AFFC0
// Address: 0x25affc0
//
__int64 __fastcall sub_25AFFC0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int8 *v4; // rbx
  int v5; // eax
  unsigned int v7; // r15d
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // r12d
  __int64 v18; // rdx
  __int64 v19; // rbx
  unsigned int v20; // ebx
  __int64 v21; // rdx
  char v22; // al
  unsigned int v23; // ecx
  unsigned int v24; // eax
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-38h]

  v4 = *(unsigned __int8 **)(a2 + 24);
  v5 = *v4;
  if ( (unsigned __int8)v5 <= 0x1Cu )
    return 0;
  v7 = a4;
  if ( (_BYTE)v5 == 30 )
  {
    v18 = a4;
    v11 = *(_QWORD *)(*((_QWORD *)v4 + 5) + 72LL);
    if ( a4 != -1 )
      return sub_25AFF40(a1, v11, v18, a3);
    v20 = 0;
    v16 = 1;
    v21 = **(_QWORD **)(*(_QWORD *)(v11 + 24) + 16LL);
    v22 = *(_BYTE *)(v21 + 8);
    if ( v22 != 7 )
    {
      do
      {
        if ( v22 == 15 )
        {
          v23 = *(_DWORD *)(v21 + 12);
        }
        else
        {
          v23 = 1;
          if ( v22 == 16 )
            v23 = *(_DWORD *)(v21 + 32);
        }
        if ( v23 <= v20 )
          break;
        v24 = sub_25AFF40(a1, v11, v20, a3);
        if ( v16 )
          v16 = v24;
        ++v20;
        v21 = **(_QWORD **)(*(_QWORD *)(v11 + 24) + 16LL);
        v22 = *(_BYTE *)(v21 + 8);
      }
      while ( v22 != 7 );
      return v16;
    }
    return 1;
  }
  if ( (_BYTE)v5 == 94 )
  {
    if ( (unsigned int)sub_BD2910(a2) )
      v7 = **((_DWORD **)v4 + 9);
    v19 = *((_QWORD *)v4 + 2);
    if ( v19 )
    {
      while ( 1 )
      {
        v16 = sub_25AFFC0(a1, v19, a3, v7);
        if ( !v16 )
          break;
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          return v16;
      }
      return 0;
    }
    return 1;
  }
  v9 = (unsigned int)(v5 - 34);
  if ( (unsigned __int8)v9 > 0x33u )
    return 0;
  v10 = 0x8000000000041LL;
  if ( !_bittest64(&v10, v9) )
    return 0;
  v11 = *((_QWORD *)v4 - 4);
  if ( !v11 )
    return 0;
  if ( *(_BYTE *)v11 )
    return 0;
  v12 = *(_QWORD *)(v11 + 24);
  if ( v12 != *((_QWORD *)v4 + 10) )
    return 0;
  if ( (v4[7] & 0x80u) == 0 )
    goto LABEL_13;
  v30 = *((_QWORD *)v4 - 4);
  v13 = sub_BD2BC0((__int64)v4);
  v11 = v30;
  if ( (v4[7] & 0x80u) == 0 )
  {
    v15 = (a2 - (__int64)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)]) >> 5;
    goto LABEL_45;
  }
  v11 = v30;
  if ( (unsigned int)((v13 + v14 - sub_BD2BC0((__int64)v4)) >> 4) )
  {
    v25 = (a2 - (__int64)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)]) >> 5;
    LODWORD(v15) = v25;
    if ( (v4[7] & 0x80u) != 0 )
    {
      v26 = sub_BD2BC0((__int64)v4);
      v11 = v30;
      if ( (v4[7] & 0x80u) == 0 )
      {
        if ( !(unsigned int)((v26 + v27) >> 4) )
          goto LABEL_12;
      }
      else
      {
        v11 = v30;
        if ( !(unsigned int)((v26 + v27 - sub_BD2BC0((__int64)v4)) >> 4) )
          goto LABEL_12;
        if ( (v4[7] & 0x80u) != 0 )
        {
          v11 = v30;
          if ( *(_DWORD *)(sub_BD2BC0((__int64)v4) + 8) <= (unsigned int)v25 )
          {
            if ( (v4[7] & 0x80u) == 0 )
              BUG();
            v28 = sub_BD2BC0((__int64)v4);
            if ( *(_DWORD *)(v28 + v29 - 4) > (unsigned int)v25 )
              return 0;
            v11 = v30;
          }
          goto LABEL_12;
        }
      }
      BUG();
    }
LABEL_45:
    v12 = *(_QWORD *)(v11 + 24);
    goto LABEL_14;
  }
LABEL_12:
  v12 = *(_QWORD *)(v11 + 24);
LABEL_13:
  v15 = (a2 - (__int64)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)]) >> 5;
LABEL_14:
  if ( *(_DWORD *)(v12 + 12) - 1 <= (unsigned int)v15 )
    return 0;
  v18 = (unsigned int)v15 | 0x100000000LL;
  return sub_25AFF40(a1, v11, v18, a3);
}
