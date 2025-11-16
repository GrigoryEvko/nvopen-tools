// Function: sub_251C230
// Address: 0x251c230
//
__int64 __fastcall sub_251C230(__int64 a1, __int64 *a2, __int64 a3, _QWORD *a4, _BYTE *a5, char a6, int a7)
{
  unsigned int v7; // r15d
  __int64 v11; // rdx
  unsigned __int8 *v12; // rax
  int v13; // eax
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  _BYTE *v17; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // r12
  unsigned int v22; // ebx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // [rsp-10h] [rbp-60h]
  _QWORD *v25; // [rsp+0h] [rbp-50h]
  _QWORD *v26; // [rsp+0h] [rbp-50h]
  unsigned __int64 v28[8]; // [rsp+10h] [rbp-40h] BYREF

  v7 = *(unsigned __int8 *)(a1 + 4300);
  if ( !(_BYTE)v7 )
    return 0;
  v11 = *a2 & 3;
  if ( v11 != 3 )
  {
    if ( v11 == 2
      || (v12 = (unsigned __int8 *)(*a2 & 0xFFFFFFFFFFFFFFFCLL)) != 0
      && (v13 = *v12, (_BYTE)v13 != 22)
      && (_BYTE)v13
      && ((unsigned __int8)v13 <= 0x1Cu
       || (v14 = (unsigned int)(v13 - 34), (unsigned __int8)v14 > 0x33u)
       || (v15 = 0x8000000000041LL, !_bittest64(&v15, v14))) )
    {
      v26 = a4;
      v17 = (_BYTE *)sub_250D070(a2);
      a4 = v26;
      if ( *v17 <= 0x15u )
        return 0;
    }
  }
  v25 = a4;
  v16 = sub_2509740(a2);
  if ( !v16 )
  {
    if ( a6 )
      return 0;
    if ( (unsigned __int8)sub_2509800(a2) != 5 )
      goto LABEL_18;
LABEL_28:
    v23 = sub_250D070(a2);
    sub_250D230(v28, v23, 3, 0);
    v20 = v28[0];
    v21 = sub_251BBC0(a1, v28[0], v28[1], a3, 2, 0, 1);
    goto LABEL_19;
  }
  if ( a6 )
    return (unsigned int)sub_251BFD0(a1, v16, a3, v25, a5, 1, a7, 0);
  if ( (unsigned __int8)sub_251BFD0(a1, v16, a3, v25, a5, 1, 1, 0) )
    return v7;
  if ( (unsigned __int8)sub_2509800(a2) == 5 )
    goto LABEL_28;
LABEL_18:
  v19 = sub_251BBC0(a1, *a2, a2[1], a3, 2, 0, 1);
  v20 = v24;
  v21 = v19;
LABEL_19:
  if ( !v21 )
    return 0;
  if ( a3 == v21 )
    return 0;
  v22 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v21 + 112LL))(v21, v20);
  if ( !(_BYTE)v22 )
    return 0;
  if ( a3 )
    sub_250ED80(a1, v21, a3, a7);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 120LL))(v21) )
  {
    v7 = v22;
    *a5 = 1;
  }
  return v7;
}
