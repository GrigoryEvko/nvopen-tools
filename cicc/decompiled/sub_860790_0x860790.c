// Function: sub_860790
// Address: 0x860790
//
__int64 __fastcall sub_860790(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  int v3; // r14d
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r8
  char v9; // al
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int8 v12; // di
  _BOOL4 v13; // eax
  __int64 v14; // rdx
  unsigned __int8 v15; // di
  _BOOL4 v16; // eax
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v2 = a2[21];
  v3 = sub_736990((__int64)a2);
  result = *(_QWORD *)(v2 + 152);
  if ( !result || (*(_BYTE *)(result + 29) & 0x20) != 0 )
    return result;
  v5 = *(_QWORD *)(result + 144);
  v6 = *(_QWORD *)(result + 112);
  v7 = *(_QWORD *)(result + 104);
  while ( v5 )
  {
    if ( (*(_BYTE *)(v5 + 193) & 0x10) != 0 )
      goto LABEL_5;
    result = *(unsigned __int8 *)(v5 + 206);
    if ( (result & 0x10) != 0 )
      goto LABEL_5;
    v8 = *(_QWORD *)v5;
    if ( v3 )
    {
      if ( (*(_BYTE *)(v5 + 88) & 4) != 0 && !*(_QWORD *)(v5 + 280) || (*(_WORD *)(v5 + 192) & 0x100A) == 2 )
      {
        v19 = *(_QWORD *)v5;
        v13 = sub_860410(v5);
        v8 = v19;
        if ( !v13 )
        {
          v14 = *a2;
          if ( dword_4D04964 || !v14 || (v15 = 4, (*(_BYTE *)(v14 + 81) & 1) != 0) )
            v15 = HIDWORD(qword_4F077B4) == 0 ? 7 : 5;
          sub_6853B0(v15, (*(_BYTE *)(v5 + 192) & 2) == 0 ? 114 : 1380, (FILE *)(v5 + 64), v19);
          v8 = v19;
        }
      }
      else
      {
        if ( (*(_BYTE *)(v8 + 81) & 1) != 0 )
        {
          result = (unsigned int)dword_4D04438;
          if ( dword_4D04438 )
            goto LABEL_22;
          goto LABEL_5;
        }
        if ( (*(_WORD *)(v5 + 192) & 0x1002) != 0 || (result & 8) != 0 || (*(_BYTE *)(v5 + 195) & 8) != 0 )
          goto LABEL_5;
        v9 = *(_BYTE *)(v5 + 174);
        if ( (unsigned __int8)(v9 - 1) > 1u && (v9 != 5 || *(_BYTE *)(v5 + 176) != 15)
          || (v20 = *(_QWORD *)v5, v16 = sub_860410(v5), v8 = v20, v16) )
        {
          if ( (*(_BYTE *)(*(_QWORD *)(*a2 + 88LL) + 177LL) & 0x10) == 0 || (*(_BYTE *)(*a2 + 81LL) & 1) == 0 )
          {
            v17 = v8;
            sub_85B2D0(v8, 0xB1u, 5);
            v8 = v17;
          }
        }
      }
    }
    result = (__int64)&dword_4D04438;
    if ( dword_4D04438 && (*(_BYTE *)(v8 + 81) & 1) != 0 )
    {
LABEL_22:
      if ( *(_BYTE *)(v5 + 172) == 1 )
      {
        result = *(unsigned __int8 *)(v5 + 195);
        if ( (result & 8) == 0 )
        {
          if ( (result & 1) == 0 || (v18 = v8, result = sub_899F90(v8), v8 = v18, !(_DWORD)result) )
            result = (__int64)sub_649830(v8, v8 + 48, 0);
        }
      }
    }
LABEL_5:
    v5 = *(_QWORD *)(v5 + 112);
  }
  while ( v6 )
  {
    v10 = *(_QWORD *)v6;
    if ( v3 )
    {
      result = *(unsigned __int8 *)(v10 + 81);
      if ( (result & 1) != 0 )
      {
        if ( (*(_BYTE *)(v6 + 169) & 0x10) == 0 && *(char *)(v10 + 84) >= 0 )
          goto LABEL_36;
        if ( !(result & 2 | *(_BYTE *)(v6 + 172) & 4) )
        {
          if ( !HIDWORD(qword_4F077B4) )
          {
            v12 = 7;
            goto LABEL_60;
          }
          v11 = *(_QWORD *)(v10 + 104);
          if ( !v11 || !*(_QWORD *)(v11 + 8) )
          {
            v12 = 5;
LABEL_60:
            sub_6853B0(v12, 0x72u, (FILE *)(v6 + 64), *(_QWORD *)v6);
          }
        }
      }
      else
      {
        sub_85B2D0(*(_QWORD *)v6, 0xB1u, 5);
      }
    }
    result = (__int64)&dword_4D04438;
    if ( dword_4D04438 && ((*(_BYTE *)(v6 + 169) & 0x10) != 0 || *(char *)(v10 + 84) < 0) && *(_BYTE *)(v6 + 136) == 1 )
    {
      if ( (*(_BYTE *)(v6 + 170) & 0x70) != 0x10 || (result = sub_899F90(v10), !(_DWORD)result) )
        result = (__int64)sub_649830(v10, v10 + 48, 0);
    }
LABEL_36:
    v6 = *(_QWORD *)(v6 + 112);
  }
  for ( ; v7; v7 = *(_QWORD *)(v7 + 112) )
  {
    if ( *(_QWORD *)v7 )
    {
      result = (unsigned int)*(unsigned __int8 *)(v7 + 140) - 9;
      if ( (unsigned __int8)(*(_BYTE *)(v7 + 140) - 9) <= 2u )
        result = sub_860790(*(unsigned __int8 *)(*(_QWORD *)v7 + 80LL), *(_QWORD *)(*(_QWORD *)v7 + 88LL));
    }
  }
  return result;
}
