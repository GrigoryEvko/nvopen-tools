// Function: sub_815C30
// Address: 0x815c30
//
unsigned __int64 __fastcall sub_815C30(__int64 a1)
{
  __int64 v1; // r12
  char v2; // al
  unsigned __int64 result; // rax
  bool v4; // zf
  char v5; // al
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  int v11; // r14d
  unsigned __int8 v12; // al
  __int64 v13; // [rsp+0h] [rbp-70h] BYREF
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+10h] [rbp-60h]
  __int64 v16; // [rsp+18h] [rbp-58h]
  char v17; // [rsp+20h] [rbp-50h]
  __int64 v18; // [rsp+28h] [rbp-48h]
  __int64 v19; // [rsp+30h] [rbp-40h]
  int i; // [rsp+38h] [rbp-38h]
  char v21; // [rsp+3Ch] [rbp-34h]
  __int64 v22; // [rsp+40h] [rbp-30h]

  v1 = a1;
  v2 = *(_BYTE *)(a1 + 89);
  v17 = 0;
  v13 = 0;
  result = v2 & 8;
  v4 = *(_QWORD *)(a1 + 8) == 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v18 = 0;
  v19 = 0;
  i = 0;
  v21 = 0;
  v22 = 0;
  if ( !v4 )
  {
    if ( (_BYTE)result )
      return result;
    goto LABEL_8;
  }
  if ( (_BYTE)result )
    return result;
  v5 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v5 - 9) <= 2u )
  {
    if ( *(_QWORD *)a1 && !*(_BYTE *)(*(_QWORD *)(a1 + 168) + 113LL) )
      goto LABEL_8;
    goto LABEL_25;
  }
  if ( v5 == 2 )
  {
    if ( (*(_BYTE *)(a1 + 161) & 8) == 0 || *(_QWORD *)a1 )
      goto LABEL_8;
LABEL_25:
    result = (unsigned __int64)sub_80AEF0(a1);
LABEL_26:
    if ( (*(_BYTE *)(a1 + 89) & 8) != 0 || (_DWORD)v19 )
      return result;
    goto LABEL_8;
  }
  if ( v5 == 14 && *(_BYTE *)(a1 + 160) == 1 )
  {
    result = sub_815F90(a1, &v13);
    goto LABEL_26;
  }
LABEL_8:
  v6 = 6;
  if ( (unsigned int)sub_80AFC0(a1, 6) || (v7 = dword_4D04440) != 0 && (v6 = 6, (unsigned int)sub_80A630(a1, 6)) )
  {
    LOBYTE(result) = *(_BYTE *)(a1 + 140);
  }
  else
  {
    result = *(unsigned __int8 *)(a1 + 140);
    if ( *(_QWORD *)(a1 + 8) )
    {
      if ( (_BYTE)result == 12 )
      {
        if ( *(_BYTE *)(a1 + 184) != 10 )
          return result;
      }
      else
      {
        if ( (unsigned __int8)(result - 9) > 2u )
          return result;
        v7 = *(_QWORD *)(a1 + 168);
        if ( !*(_QWORD *)(v7 + 168) )
          return result;
      }
    }
  }
  v11 = 2;
  if ( (unsigned __int8)(result - 9) > 2u )
    goto LABEL_15;
LABEL_11:
  v12 = *(_BYTE *)(v1 + 177);
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  for ( i = v12 >> 7; ; i = 0 )
  {
    v21 = 0;
    v22 = 0;
    sub_809110(a1, v6, v7, v8, v9, v10);
    sub_823800(qword_4F18BE0);
    if ( v11 == 1 )
      break;
    v13 += 2;
    sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
    sub_810650(v1, 1, &v13);
    v6 = 0;
    a1 = v1;
    sub_80B290(v1, 0, (__int64)&v13);
    result = (unsigned int)v22;
    if ( !(_DWORD)v22 )
      return result;
    v11 = 1;
    if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) <= 2u )
      goto LABEL_11;
LABEL_15:
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v19 = 0;
  }
  v13 += 2;
  HIDWORD(v22) = 1;
  sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
  sub_810650(v1, 1, &v13);
  return (unsigned __int64)sub_80B290(v1, 0, (__int64)&v13);
}
