// Function: sub_829A30
// Address: 0x829a30
//
__int64 __fastcall sub_829A30(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rax
  char v8; // dl
  __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 *v12; // rdi
  _QWORD *v13; // [rsp+8h] [rbp-28h] BYREF
  int v14; // [rsp+14h] [rbp-1Ch] BYREF
  __int64 *v15; // [rsp+18h] [rbp-18h] BYREF

  v3 = *(_QWORD *)(a1 + 88);
  v13 = a2;
  v4 = *(_QWORD *)(v3 + 88);
  if ( v4 && (*(_BYTE *)(v3 + 160) & 1) == 0 )
    v3 = *(_QWORD *)(v4 + 88);
  else
    v4 = a1;
  v5 = *(_QWORD *)(v3 + 176);
  v15 = 0;
  v14 = 0;
  v6 = *(_QWORD *)(v5 + 88);
  if ( (*(_BYTE *)(v3 + 265) & 1) != 0 )
  {
    v7 = *(_QWORD *)(v6 + 160);
    v8 = *(_BYTE *)(v7 + 140);
    if ( (unsigned __int8)(v8 - 9) > 2u )
    {
      if ( v8 != 12 || *(_BYTE *)(v7 + 184) != 10 )
        return 0;
      v10 = **(_QWORD **)(*(_QWORD *)(v7 + 168) + 16LL);
    }
    else
    {
      if ( (*(_BYTE *)(v7 + 177) & 0x10) == 0 )
        return 0;
      v10 = **(_QWORD **)(*(_QWORD *)(v7 + 168) + 160LL);
    }
    if ( !v10 || (*(_BYTE *)(v10 + 81) & 2) == 0 || *(char *)(*(_QWORD *)(v10 + 88) + 266LL) < 0 )
      return 0;
    v11 = sub_829A30(v10, v13);
    if ( !v11 )
      goto LABEL_26;
    result = sub_8C0950(v4, v11);
  }
  else
  {
    if ( (unsigned __int8)(*(_BYTE *)(v6 + 140) - 9) > 2u || (*(_BYTE *)(*(_QWORD *)(v5 + 96) + 178LL) & 0x40) == 0 )
      return 0;
    if ( !(unsigned int)sub_8294E0(&v15, v6, &v13, &v14) )
    {
LABEL_26:
      v12 = v15;
      if ( !v15 )
        return 0;
      goto LABEL_27;
    }
    v12 = v15;
    if ( !v15 )
      return 0;
    if ( v13 && !v14 )
    {
LABEL_27:
      sub_724F80(v12);
      return 0;
    }
    result = sub_8C2730(a1, v15);
  }
  if ( !result )
    goto LABEL_26;
  return result;
}
