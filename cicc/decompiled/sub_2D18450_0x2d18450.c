// Function: sub_2D18450
// Address: 0x2d18450
//
__int64 __fastcall sub_2D18450(
        __int64 a1,
        __int16 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v9; // r13d
  _QWORD *v11; // r15
  _QWORD *v12; // r14
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  int v19; // esi
  _QWORD *v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rdx
  _DWORD v23[13]; // [rsp+14h] [rbp-34h] BYREF

  v23[0] = 0;
  v9 = sub_C55A30(a1 + 216, a1, a4, a5, a7, a8, v23);
  if ( !(_BYTE)v9 )
  {
    *(_QWORD *)(a1 + 144) -= 4LL;
    *(_QWORD *)(a1 + 200) -= 4LL;
    *(_WORD *)(a1 + 14) = a2;
    v11 = sub_C52410();
    v12 = v11 + 1;
    v13 = sub_C959E0();
    v14 = (_QWORD *)v11[2];
    if ( v14 )
    {
      v15 = v11 + 1;
      do
      {
        while ( 1 )
        {
          v16 = v14[2];
          v17 = v14[3];
          if ( v13 <= v14[4] )
            break;
          v14 = (_QWORD *)v14[3];
          if ( !v17 )
            goto LABEL_8;
        }
        v15 = v14;
        v14 = (_QWORD *)v14[2];
      }
      while ( v16 );
LABEL_8:
      if ( v12 != v15 && v13 >= v15[4] )
        v12 = v15;
    }
    if ( v12 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_19;
    v18 = v12[7];
    if ( !v18 )
      goto LABEL_19;
    v19 = *(_DWORD *)(a1 + 8);
    v20 = v12 + 6;
    do
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(v18 + 16);
        v22 = *(_QWORD *)(v18 + 24);
        if ( *(_DWORD *)(v18 + 32) >= v19 )
          break;
        v18 = *(_QWORD *)(v18 + 24);
        if ( !v22 )
          goto LABEL_17;
      }
      v20 = (_QWORD *)v18;
      v18 = *(_QWORD *)(v18 + 16);
    }
    while ( v21 );
LABEL_17:
    if ( v12 + 6 == v20 || v19 < *((_DWORD *)v20 + 8) )
LABEL_19:
      sub_2D18210((int *)(a1 + 8), 0xFFFFFFFF);
    else
      sub_2D18210((int *)(a1 + 8), *((_DWORD *)v20 + 9) - 1);
  }
  return v9;
}
