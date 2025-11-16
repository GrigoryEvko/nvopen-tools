// Function: sub_11DCF00
// Address: 0x11dcf00
//
__int64 __fastcall sub_11DCF00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, unsigned int **a6)
{
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // r14
  __int64 v13; // rsi
  unsigned __int8 v14; // al
  unsigned __int8 v15; // dl
  _BYTE *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v20; // [rsp+8h] [rbp-88h]
  __int64 v21; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v24; // [rsp+28h] [rbp-68h]
  _BYTE v25[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v26; // [rsp+50h] [rbp-40h]

  if ( a2 == a3 )
    return sub_AD6530(*(_QWORD *)(a1 + 8), a2);
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  if ( !(unsigned __int8)sub_98B0F0(a2, &v21, 0) || !(unsigned __int8)sub_98B0F0(a3, &v23, 0) )
    return 0;
  v10 = sub_AD64C0(*(_QWORD *)(a1 + 8), 0, 0);
  v11 = v22;
  if ( v24 <= v22 )
    v11 = v24;
  v12 = v10;
  if ( v11 )
  {
    v13 = 0;
    while ( 1 )
    {
      v14 = *(_BYTE *)(v21 + v13);
      v15 = *(_BYTE *)(v23 + v13);
      if ( a5 )
      {
        if ( !v14 )
          break;
      }
      if ( v15 != v14 )
        goto LABEL_11;
      if ( ++v13 == v11 )
        return v12;
    }
    if ( !v15 )
      return v12;
LABEL_11:
    v20 = -(__int64)(v14 < v15) | 1;
    v16 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a4 + 8), v13, 0);
    v26 = 257;
    v17 = sub_92B530(a6, 0x25u, a4, v16, (__int64)v25);
    v18 = sub_AD64C0(*(_QWORD *)(a1 + 8), v20, 0);
    v26 = 257;
    return sub_B36550(a6, v17, v12, v18, (__int64)v25, 0);
  }
  return v12;
}
