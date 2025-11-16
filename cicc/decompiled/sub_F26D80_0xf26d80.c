// Function: sub_F26D80
// Address: 0xf26d80
//
__int64 __fastcall sub_F26D80(__int64 a1, __int64 *a2, unsigned __int8 *a3)
{
  int v6; // edx
  __int64 *v7; // r14
  __int64 v8; // r15
  __int64 **v9; // rax
  __int64 v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // r9
  _QWORD *v13; // rbx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rsi
  __int64 result; // rax
  __int64 *v21; // rbx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // rdi
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *a3;
  if ( (unsigned int)(v6 - 12) <= 1 )
  {
    v7 = (__int64 *)sub_BD5C60((__int64)a2);
    v8 = sub_ACD6D0(v7);
    v9 = (__int64 **)sub_BCE3C0(v7, 0);
    v10 = sub_ACADE0(v9);
    v11 = sub_BD2C40(80, unk_3F10A10);
    v13 = v11;
    if ( v11 )
      sub_B4D3C0((__int64)v11, v8, v10, 0, 0, v12, 0, 0);
    v14 = a2[6];
    v27[0] = v14;
    if ( v14 )
    {
      sub_B96E90((__int64)v27, v14, 1);
      v15 = v13[6];
      v16 = (__int64)(v13 + 6);
      if ( !v15 )
        goto LABEL_7;
    }
    else
    {
      v15 = v13[6];
      v16 = (__int64)(v13 + 6);
      if ( !v15 )
      {
LABEL_9:
        sub_B44220(v13, (__int64)(a2 + 3), 0);
        v18 = *(_QWORD *)(a1 + 40);
        v27[0] = (__int64)v13;
        sub_F200C0(v18 + 2096, v27);
LABEL_10:
        v19 = a2;
        return sub_F207A0(a1, v19);
      }
    }
    v26 = v16;
    sub_B91220(v16, v15);
    v16 = v26;
LABEL_7:
    v17 = (unsigned __int8 *)v27[0];
    v13[6] = v27[0];
    if ( v17 )
      sub_B976B0((__int64)v27, v17, v16);
    goto LABEL_9;
  }
  if ( (_BYTE)v6 == 20 )
    goto LABEL_10;
  if ( (_BYTE)v6 == 85 )
  {
    v23 = *((_QWORD *)a3 + 2);
    if ( v23 )
    {
      if ( !*(_QWORD *)(v23 + 8) )
      {
        v24 = sub_D5CCF0(a3);
        if ( v24 )
        {
          v19 = (__int64 *)sub_F162A0(a1, (__int64)a3, v24);
          return sub_F207A0(a1, v19);
        }
      }
    }
  }
  result = 0;
  if ( *(_BYTE *)(a1 + 48) )
  {
    v21 = *(__int64 **)(a1 + 72);
    if ( !(unsigned __int8)sub_A73ED0(a2 + 9, 23) && !(unsigned __int8)sub_B49560((__int64)a2, 23)
      || (unsigned __int8)sub_A73ED0(a2 + 9, 4)
      || (unsigned __int8)sub_B49560((__int64)a2, 4) )
    {
      v22 = *(a2 - 4);
      if ( !v22 )
        return 0;
      if ( *(_BYTE *)v22 )
        return 0;
      if ( *(_QWORD *)(v22 + 24) != a2[10] )
        return 0;
      if ( !sub_981210(*v21, v22, (unsigned int *)v27) )
        return 0;
      v25 = *(_QWORD **)(a1 + 72);
      if ( (v25[((unsigned __int64)LODWORD(v27[0]) >> 6) + 1] & (1LL << SLOBYTE(v27[0]))) != 0 )
        return 0;
      if ( (((int)*(unsigned __int8 *)(*v25 + (LODWORD(v27[0]) >> 2)) >> (2 * (v27[0] & 3))) & 3) == 0 )
        return 0;
      if ( LODWORD(v27[0]) != 289 )
        return 0;
      result = sub_F094D0((__int64)a2, *(_BYTE **)(a1 + 88));
      if ( !result )
        return 0;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
