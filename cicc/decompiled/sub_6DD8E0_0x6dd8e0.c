// Function: sub_6DD8E0
// Address: 0x6dd8e0
//
__int64 __fastcall sub_6DD8E0(__int64 *a1, _QWORD *a2, __int64 a3, _QWORD *a4)
{
  int v8; // eax
  __int64 v9; // rdi
  int v10; // r10d
  _QWORD *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // r10d
  _QWORD *v16; // rdi
  __int64 v17; // r12
  int v19; // eax
  int v20; // edx
  int v21; // edx
  __int64 v22; // rax
  char i; // dl
  __int64 v24; // [rsp-8h] [rbp-278h]
  int v25; // [rsp+Ch] [rbp-264h]
  _QWORD v26[2]; // [rsp+10h] [rbp-260h] BYREF
  unsigned int v27; // [rsp+20h] [rbp-250h]
  __int64 v28; // [rsp+24h] [rbp-24Ch]
  __int64 v29; // [rsp+30h] [rbp-240h]
  _BYTE v30[18]; // [rsp+40h] [rbp-230h] BYREF
  __int16 v31; // [rsp+52h] [rbp-21Eh]
  _QWORD v32[2]; // [rsp+E0h] [rbp-190h] BYREF
  char v33; // [rsp+F0h] [rbp-180h]
  char v34; // [rsp+F1h] [rbp-17Fh]

  if ( unk_4F074B0 && (unsigned int)sub_732350(a1) )
  {
    *(_BYTE *)(a3 + 56) = 1;
    sub_6E6260(v32);
    return sub_6F6F40(v32, 0);
  }
  sub_68A670(a3, (__int64)v26);
  v8 = *(_DWORD *)(a3 + 40);
  v9 = 5;
  v10 = v8 & 0x80;
  if ( (v8 & 0x80) != 0 )
  {
    LOBYTE(v8) = v8 & 0x7F;
    v10 = 1;
    v9 = 2;
    *(_DWORD *)(a3 + 40) = v8;
  }
  v25 = v10;
  sub_6E2140(v9, v30, 0, 0, a3);
  v11 = (_QWORD *)a3;
  v31 |= 0x2C0u;
  sub_6DC380(a1, a3, 0, v32, 0, 1u);
  v15 = v25;
  if ( *(_BYTE *)(a3 + 56) )
    goto LABEL_6;
  if ( a2 )
  {
    if ( v33 == 3 )
    {
      if ( !v25 )
      {
        v11 = v32;
        sub_6FC070(a2, v32, (*(_DWORD *)(a3 + 40) >> 15) & 1, 0, 0);
        goto LABEL_19;
      }
      v15 = 4;
      goto LABEL_15;
    }
    if ( word_4D04898 | v25 )
    {
      if ( v25 )
        v15 = 4;
LABEL_15:
      v19 = *(_DWORD *)(a3 + 40);
      if ( (v19 & 0x1000) != 0 )
      {
        v20 = v15;
        v15 |= 0x1000008u;
        v21 = v20 | 8;
        if ( (v19 & 0x8000) == 0 )
          v15 = v21;
      }
      v11 = a2;
      sub_843C40((unsigned int)v32, (_DWORD)a2, 0, 0, 1, v15, 458);
      v12 = v24;
      goto LABEL_19;
    }
  }
  if ( v34 != 3 )
  {
LABEL_20:
    if ( v33 == 2 )
    {
      v11 = a4;
      v16 = v32;
      v17 = 0;
      sub_6F4950(v32, a4, v12, 2, v13, v14);
    }
    else
    {
      if ( !v33 )
        goto LABEL_30;
      v22 = v32[0];
      for ( i = *(_BYTE *)(v32[0] + 140LL); i == 12; i = *(_BYTE *)(v22 + 140) )
        v22 = *(_QWORD *)(v22 + 160);
      if ( i )
      {
        if ( v33 != 1 )
        {
          *(_BYTE *)(a3 + 56) = 1;
          sub_6E6840(v32);
        }
        v11 = 0;
        v16 = v32;
        v17 = sub_6F6F40(v32, 0);
      }
      else
      {
LABEL_30:
        sub_6E6870(v32);
        v11 = 0;
        v16 = v32;
        v17 = sub_6F6F40(v32, 0);
      }
    }
    goto LABEL_7;
  }
  v11 = 0;
  sub_6F6890(v32, 0);
LABEL_19:
  if ( !*(_BYTE *)(a3 + 56) )
    goto LABEL_20;
LABEL_6:
  v16 = a4;
  v17 = 0;
  sub_72C970(a4);
LABEL_7:
  sub_6E2B30(v16, v11);
  if ( v29 )
  {
    sub_878D40();
    sub_6E1DF0(v26[0]);
    qword_4F06BC0 = v26[1];
    *(_QWORD *)&dword_4F061D8 = v28;
    sub_729730(v27);
  }
  return v17;
}
