// Function: sub_2FF6F50
// Address: 0x2ff6f50
//
unsigned __int64 __fastcall sub_2FF6F50(__int64 a1, signed int a2, __int64 a3)
{
  unsigned __int64 v5; // r8
  bool v6; // cl
  bool v7; // dl
  unsigned __int8 v8; // r9
  char v9; // al
  char v10; // cl
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // r8
  int v18; // edx
  int v19; // eax
  __int64 v20; // rax
  _BYTE *v21; // rax
  __int64 *v22; // r8
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax

  if ( (unsigned int)(a2 - 1) <= 0x3FFFFFFE )
  {
    v22 = sub_2FF6500(a1, a2, 1);
    v19 = *(_DWORD *)(a1 + 328) * ((__int64)(*(_QWORD *)(a1 + 288) - *(_QWORD *)(a1 + 280)) >> 3);
    v18 = *(unsigned __int16 *)(*v22 + 24);
    return *(unsigned int *)(*(_QWORD *)(a1 + 312) + 16LL * (unsigned int)(v18 + v19));
  }
  if ( a2 >= 0 || (v20 = a2 & 0x7FFFFFFF, (unsigned int)v20 >= *(_DWORD *)(a3 + 464)) )
  {
    v5 = 0;
    v6 = 0;
    v7 = 0;
    v8 = 0;
  }
  else
  {
    v21 = (_BYTE *)(*(_QWORD *)(a3 + 456) + 8 * v20);
    v8 = *v21 & 1;
    v6 = (*v21 & 4) != 0;
    v5 = *(_QWORD *)v21 >> 3;
    v7 = (*v21 & 2) != 0;
  }
  v9 = (8 * v5) | (4 * v6) | v8 | (2 * v7);
  if ( !((8 * v5) | (unsigned __int16)((4 * v6) | v8 | (unsigned __int16)(2 * v7)) & 0xFFF9) )
  {
    v18 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 56) + 16LL * (a2 & 0x7FFFFFFF))
                                          & 0xFFFFFFFFFFFFFFF8LL)
                              + 24LL);
    v19 = *(_DWORD *)(a1 + 328) * ((__int64)(*(_QWORD *)(a1 + 288) - *(_QWORD *)(a1 + 280)) >> 3);
    return *(unsigned int *)(*(_QWORD *)(a1 + 312) + 16LL * (unsigned int)(v18 + v19));
  }
  v10 = (v8 | (2 * v7)) & 2;
  if ( (v9 & 6) == 2 || (v9 & 1) != 0 )
  {
    v23 = v5;
    v16 = v5 >> 29;
    v24 = v23 >> 45;
    if ( v10 )
      return v24;
  }
  else
  {
    v11 = v5;
    v12 = v5;
    v13 = v5 >> 29;
    v14 = v11 >> 5;
    v15 = v12 >> 45;
    if ( v10 )
      LODWORD(v13) = v15;
    return (unsigned __int16)v14 * (unsigned int)v13;
  }
  return v16;
}
