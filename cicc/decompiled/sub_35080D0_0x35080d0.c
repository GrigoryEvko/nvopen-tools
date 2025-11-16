// Function: sub_35080D0
// Address: 0x35080d0
//
__int64 __fastcall sub_35080D0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v5; // edi
  __int64 v6; // r11
  unsigned int v7; // eax
  __int64 v8; // r8
  unsigned __int16 *v9; // rdx
  unsigned int v10; // r8d
  char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r10
  char *v15; // r8
  unsigned int v16; // edi
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // rcx
  _WORD *v20; // rdx

  v5 = (unsigned __int16)a3;
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 48) + (unsigned __int16)a3);
  if ( v7 >= (unsigned int)v6 )
    goto LABEL_8;
  v8 = *(_QWORD *)(a1 + 8);
  while ( 1 )
  {
    v9 = (unsigned __int16 *)(v8 + 2LL * v7);
    if ( *v9 == v5 )
      break;
    v7 += 256;
    if ( (unsigned int)v6 <= v7 )
      goto LABEL_8;
  }
  if ( v9 == (unsigned __int16 *)(v8 + 2 * v6) )
  {
LABEL_8:
    v10 = 0;
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 384) + 8LL * (a3 >> 6)) & (1LL << a3)) != 0 )
      return v10;
    v12 = sub_E922F0(*(_QWORD **)a1, a3);
    v14 = (__int64)&v12[2 * v13 - 2];
    v15 = v12;
    if ( v12 == (char *)v14 )
      return 1;
    v16 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v17 = *(unsigned __int16 *)v15;
      v18 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 48) + v17);
      if ( v18 < v16 )
      {
        v19 = *(_QWORD *)(a1 + 8);
        while ( 1 )
        {
          v20 = (_WORD *)(v19 + 2LL * v18);
          if ( (_WORD)v17 == *v20 )
            break;
          v18 += 256;
          if ( v16 <= v18 )
            goto LABEL_16;
        }
        if ( v20 != (_WORD *)(2LL * *(_QWORD *)(a1 + 16) + v19) )
          break;
      }
LABEL_16:
      v15 += 2;
      if ( (char *)v14 == v15 )
        return 1;
    }
  }
  return 0;
}
