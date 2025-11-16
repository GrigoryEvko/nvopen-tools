// Function: sub_39B2440
// Address: 0x39b2440
//
__int64 __fastcall sub_39B2440(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // rax
  unsigned int v7; // r14d
  unsigned __int64 v8; // r15
  unsigned int v10; // esi
  unsigned __int8 v11; // al
  __int64 v12; // rcx
  char v13; // al
  int v14; // r14d
  int v15; // r15d
  char v16; // al
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // [rsp+0h] [rbp-40h]
  unsigned __int64 v20; // [rsp+8h] [rbp-38h]
  int v21; // [rsp+8h] [rbp-38h]

  v6 = sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), a3, a4);
  v7 = v6;
  v8 = HIDWORD(v6);
  if ( *(_BYTE *)(a3 + 8) != 16 )
    return v7;
  v20 = v6;
  v10 = sub_1643030(a3);
  if ( v10 >= (unsigned int)sub_39B1510(v8) )
    return v7;
  v11 = sub_39B1E70(*(_QWORD *)(a1 + 8), a3);
  v12 = v20;
  if ( a2 == 31 )
  {
    if ( v11 && (_BYTE)v8 )
    {
      v13 = *(_BYTE *)(v11 + *(_QWORD *)(a1 + 24) + 115LL * (unsigned __int8)v8 + 58658);
LABEL_8:
      if ( (v13 & 0xFB) == 0 )
        return v7;
    }
  }
  else if ( v11 && (_BYTE)v8 )
  {
    v13 = (unsigned __int8)*(_WORD *)(*(_QWORD *)(a1 + 24) + 2 * (v11 + 115LL * (unsigned __int8)v8 + 16104)) >> 4;
    goto LABEL_8;
  }
  v21 = *(_QWORD *)(a3 + 32);
  if ( v21 > 0 )
  {
    v14 = 0;
    v15 = 0;
    while ( 1 )
    {
      v16 = *(_BYTE *)(a3 + 8);
      if ( a2 == 31 )
      {
        v17 = a3;
        if ( v16 == 16 )
          goto LABEL_16;
      }
      else
      {
        v17 = a3;
        if ( v16 == 16 )
LABEL_16:
          v17 = **(_QWORD **)(a3 + 16);
      }
      v19 = v12;
      ++v15;
      v18 = sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v17, v12);
      v12 = v19;
      v14 += v18;
      if ( v21 == v15 )
        return (unsigned int)(v19 + v14);
    }
  }
  return v7;
}
