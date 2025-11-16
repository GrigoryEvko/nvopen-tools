// Function: sub_3506EE0
// Address: 0x3506ee0
//
unsigned __int64 __fastcall sub_3506EE0(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r14
  __int64 *v9; // r12
  unsigned __int64 result; // rax
  __int64 v11; // r13
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 i; // rsi
  int v15; // edx
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned __int64 *v18; // rdx
  int v19; // edx
  int v20; // r10d

  v7 = a1[1];
  v8 = a1[2];
  v9 = (__int64 *)a1[4];
  if ( a3 < 0 )
  {
    result = *(_QWORD *)(v7 + 56) + 16LL * (a3 & 0x7FFFFFFF);
    v11 = *(_QWORD *)(result + 8);
  }
  else
  {
    result = *(_QWORD *)(v7 + 304);
    v11 = *(_QWORD *)(result + 8LL * (unsigned int)a3);
  }
  if ( v11 )
  {
    if ( (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(v11 + 16);
        v13 = v12;
        if ( (*(_DWORD *)(v12 + 44) & 4) != 0 )
        {
          do
            v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v13 + 44) & 4) != 0 );
        }
        for ( ; (*(_BYTE *)(v12 + 44) & 8) != 0; v12 = *(_QWORD *)(v12 + 8) )
          ;
        for ( i = *(_QWORD *)(v12 + 8); i != v13; v13 = *(_QWORD *)(v13 + 8) )
        {
          v15 = *(unsigned __int16 *)(v13 + 68);
          v12 = (unsigned int)(v15 - 14);
          if ( (unsigned __int16)(v15 - 14) > 4u && (_WORD)v15 != 24 )
            break;
        }
        v16 = *(unsigned int *)(v8 + 144);
        v17 = *(_QWORD *)(v8 + 128);
        if ( !(_DWORD)v16 )
          goto LABEL_20;
        a5 = (unsigned int)(v16 - 1);
        v12 = (unsigned int)a5 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v18 = (unsigned __int64 *)(v17 + 16 * v12);
        a6 = *v18;
        if ( v13 != *v18 )
          break;
LABEL_15:
        result = sub_2E0E0B0(
                   a2,
                   ((*(_BYTE *)(v11 + 4) & 4) == 0 ? 4LL : 2LL) | v18[1] & 0xFFFFFFFFFFFFFFF8LL,
                   v9,
                   v12,
                   a5,
                   a6);
        v11 = *(_QWORD *)(v11 + 32);
        if ( !v11 )
          return result;
LABEL_16:
        if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 )
          return result;
      }
      v19 = 1;
      while ( a6 != -4096 )
      {
        v20 = v19 + 1;
        v12 = (unsigned int)a5 & (v19 + (_DWORD)v12);
        v18 = (unsigned __int64 *)(v17 + 16LL * (unsigned int)v12);
        a6 = *v18;
        if ( v13 == *v18 )
          goto LABEL_15;
        v19 = v20;
      }
LABEL_20:
      v18 = (unsigned __int64 *)(v17 + 16 * v16);
      goto LABEL_15;
    }
    v11 = *(_QWORD *)(v11 + 32);
    if ( v11 )
      goto LABEL_16;
  }
  return result;
}
