// Function: sub_2C14770
// Address: 0x2c14770
//
__int64 __fastcall sub_2C14770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int8 *v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rcx
  int v9; // esi
  __int64 result; // rax
  __int64 v11; // rbx
  unsigned __int8 **v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r12
  __int64 v15; // rax
  bool v16; // of

  v4 = *(_QWORD *)(a1 + 96);
  if ( *(_BYTE *)v4 != 61 )
    v4 = *(_QWORD *)(v4 - 64);
  v5 = sub_2AAEDF0(*(_QWORD *)(v4 + 8), a2);
  v6 = *(unsigned __int8 **)(a1 + 96);
  v7 = v5;
  sub_2AAE0E0((__int64)v6);
  v8 = *((_QWORD *)v6 - 4);
  v9 = *v6;
  if ( (_BYTE)v9 == 61 )
  {
    if ( *(_BYTE *)(a1 + 104) )
      goto LABEL_5;
LABEL_15:
    v14 = sub_DFD550(*(__int64 **)a3, (unsigned int)(v9 - 29), v7, v8, *(_BYTE *)(a1 + 106));
    v15 = sub_DFDB90(*(_QWORD *)a3);
    v16 = __OFADD__(v14, v15);
    result = v14 + v15;
    if ( v16 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v14 <= 0 )
        return 0x8000000000000000LL;
    }
    return result;
  }
  if ( !*(_BYTE *)(a1 + 104) )
  {
    if ( (_BYTE)v9 != 62 )
      v8 = 0;
    goto LABEL_15;
  }
LABEL_5:
  if ( *(_BYTE *)(a1 + 106) )
  {
    result = sub_DFD500(*(_QWORD *)a3);
    v11 = result;
    if ( !*(_BYTE *)(a1 + 105) )
      return result;
    goto LABEL_11;
  }
  if ( (v6[7] & 0x40) != 0 )
    v12 = (unsigned __int8 **)*((_QWORD *)v6 - 1);
  else
    v12 = (unsigned __int8 **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
  sub_DFB770(*v12);
  result = sub_DFD4A0(*(__int64 **)a3);
  v11 = result;
  if ( *(_BYTE *)(a1 + 105) )
  {
LABEL_11:
    v13 = sub_DFBC30(*(__int64 **)a3, 1, v7, 0, 0, *(unsigned int *)(a3 + 176), 0, 0, 0, 0, 0);
    result = v13 + v11;
    if ( __OFADD__(v13, v11) )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v13 <= 0 )
        return 0x8000000000000000LL;
    }
  }
  return result;
}
