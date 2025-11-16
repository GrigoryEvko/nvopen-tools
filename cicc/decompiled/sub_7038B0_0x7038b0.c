// Function: sub_7038B0
// Address: 0x7038b0
//
__int64 __fastcall sub_7038B0(
        unsigned __int8 a1,
        __m128i *a2,
        __m128i *a3,
        __m128i *a4,
        __int64 *a5,
        int a6,
        _QWORD *a7)
{
  __int64 v10; // r15
  __int8 v11; // al
  __int8 v12; // al
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  __int64 v27; // r9
  unsigned int v28; // [rsp+Ch] [rbp-44h]
  unsigned int v29; // [rsp+10h] [rbp-40h]

  v10 = dword_4D03B80;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
  {
    v14 = 1;
    if ( (unsigned int)sub_68FE10(a2, 1, 0) || (v14 = 0, (unsigned int)sub_68FE10(a3, 0, 0)) )
    {
      sub_6F40C0((__int64)a2, v14, v15, v16, v17, v18);
      sub_6F40C0((__int64)a3, v14, v19, v20, v21, v22);
      return sub_700E50(a1, a2, a3, v10, 0, a4, a5, a6, a7);
    }
    v28 = sub_736D90(a1);
    v29 = sub_736DB0(a1);
    sub_6F3DD0((__int64)a2, v28, v28 == 0, v23, v24, v25);
    sub_6F3DD0((__int64)a3, v29, v29 == 0, v26, v29, v27);
  }
  else
  {
    v11 = a2[1].m128i_i8[0];
    if ( v11 == 3 )
    {
      sub_6F3BA0(a2, 1);
      v12 = a3[1].m128i_i8[0];
      if ( v12 != 3 )
      {
LABEL_6:
        if ( v12 == 4 )
          sub_6EE880((__int64)a3, 0);
        sub_6F69D0(a3, 0);
        goto LABEL_9;
      }
    }
    else
    {
      if ( v11 == 4 )
        sub_6EE880((__int64)a2, 0);
      sub_6F69D0(a2, 0);
      v12 = a3[1].m128i_i8[0];
      if ( v12 != 3 )
        goto LABEL_6;
    }
    sub_6F3BA0(a3, 1);
  }
LABEL_9:
  if ( (unsigned int)sub_730FB0(a1) )
    v10 = sub_6EFF80();
  return sub_700E50(a1, a2, a3, v10, 0, a4, a5, a6, a7);
}
