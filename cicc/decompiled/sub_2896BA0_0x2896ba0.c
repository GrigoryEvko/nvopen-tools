// Function: sub_2896BA0
// Address: 0x2896ba0
//
__int64 __fastcall sub_2896BA0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned __int8 v4; // al
  __int64 v6; // r13
  void *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // r13
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v21; // [rsp+28h] [rbp-78h] BYREF
  __int64 v22; // [rsp+30h] [rbp-70h] BYREF
  __int64 v23; // [rsp+38h] [rbp-68h]
  int v24; // [rsp+40h] [rbp-60h]
  __int64 *v25; // [rsp+50h] [rbp-50h] BYREF
  __int64 v26; // [rsp+60h] [rbp-40h]

  v4 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  if ( v4 == 85 )
  {
    v17 = *(_QWORD *)(a2 - 32);
    if ( !v17
      || *(_BYTE *)v17
      || *(_QWORD *)(v17 + 24) != *(_QWORD *)(a2 + 80)
      || (*(_BYTE *)(v17 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v17 + 36) - 231) > 3 )
    {
      return 0;
    }
  }
  else if ( (unsigned int)v4 - 41 > 6 && (unsigned __int8)(v4 - 61) > 1u )
  {
    return 0;
  }
  sub_2895280(&v25, (__int64 *)(a1 + 64), a2);
  v6 = v26;
  if ( v26 != *(_QWORD *)(a1 + 72) + 24LL * *(unsigned int *)(a1 + 88) )
  {
    if ( (_BYTE)qword_5003F28 )
    {
      if ( a3 != *(_QWORD *)(v26 + 8) )
      {
        v7 = sub_CB72A0();
        v8 = sub_904010((__int64)v7, "Conflicting shapes (");
        v9 = sub_CB59D0(v8, *(unsigned int *)(v6 + 8));
        v10 = sub_904010(v9, "x");
        v11 = sub_CB59D0(v10, *(unsigned int *)(v6 + 12));
        v12 = sub_904010(v11, " vs ");
        v13 = sub_CB59D0(v12, (unsigned int)a3);
        v14 = sub_904010(v13, "x");
        v15 = sub_CB59D0(v14, HIDWORD(a3));
        v16 = (_BYTE *)sub_904010(v15, ") for ");
        sub_A69870(a2, v16, 0);
        sub_904010((__int64)v16, "\n");
        sub_C64ED0("Matrix shape verification failed, compilation aborted!", 1u);
      }
    }
    return 0;
  }
  v22 = a2;
  v23 = a3;
  v24 = a4;
  if ( !(unsigned __int8)sub_28941A0(a1 + 64, &v22, &v21) )
  {
    v18 = sub_2895620(a1 + 64, &v22, v21);
    *v18 = v22;
    v18[1] = v23;
    *((_DWORD *)v18 + 4) = v24;
  }
  return 1;
}
