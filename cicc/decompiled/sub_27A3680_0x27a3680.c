// Function: sub_27A3680
// Address: 0x27a3680
//
__int64 __fastcall sub_27A3680(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _BYTE *v6; // r8
  __int64 result; // rax
  __int64 v8; // rbx
  char v9; // al
  char v10; // al
  __int64 v11; // r8
  __int64 v12; // r8
  unsigned __int8 v13; // al
  char v14; // al
  __int64 v15; // [rsp+0h] [rbp-40h]
  _BYTE *v16; // [rsp+8h] [rbp-38h]
  unsigned __int8 v17; // [rsp+8h] [rbp-38h]
  _BYTE *v18; // [rsp+8h] [rbp-38h]

  if ( *a2 == 61 )
  {
    v6 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( !v6 )
      BUG();
    if ( *v6 != 63 )
      return 0;
LABEL_19:
    v18 = v6;
    v13 = sub_27A2DA0(a1, (__int64)v6, a3);
    if ( v13 )
    {
      v12 = (__int64)v18;
      v17 = v13;
LABEL_21:
      sub_27A2E50(a1, (__int64)a2, a3, a4, v12);
      return v17;
    }
    return 0;
  }
  if ( *a2 != 62 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)a2 - 4);
  v8 = *((_QWORD *)a2 - 8);
  if ( *v6 == 63 )
  {
    v9 = *(_BYTE *)v8;
    if ( *(_BYTE *)v8 <= 0x1Cu )
      goto LABEL_19;
  }
  else
  {
    v9 = *(_BYTE *)v8;
    if ( *(_BYTE *)v8 <= 0x1Cu )
      return 0;
    v6 = 0;
  }
  v16 = v6;
  if ( v9 == 63 )
  {
    v14 = sub_27A2DA0(a1, v8, a3);
    v11 = (__int64)v16;
    if ( !v14 )
      return 0;
  }
  else
  {
    v10 = sub_B19720(*(_QWORD *)(a1 + 216), *(_QWORD *)(v8 + 40), a3);
    v11 = (__int64)v16;
    if ( !v10 )
      return 0;
  }
  if ( !v11 )
    return 0;
  v15 = v11;
  v17 = sub_27A2DA0(a1, v11, a3);
  if ( !v17 )
    return 0;
  sub_27A2E50(a1, (__int64)a2, a3, a4, v15);
  result = v17;
  if ( v8 )
  {
    v12 = v8;
    if ( *(_BYTE *)v8 == 63 )
      goto LABEL_21;
  }
  return result;
}
