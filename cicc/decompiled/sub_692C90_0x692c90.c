// Function: sub_692C90
// Address: 0x692c90
//
__int64 __fastcall sub_692C90(__int64 a1, __int64 a2)
{
  _DWORD *v2; // r15
  __int64 v4; // r13
  __int64 i; // r12
  __int64 v6; // r11
  __int64 v7; // r15
  __int64 result; // rax
  __int64 v9; // r11
  int v10; // esi
  _BOOL4 v11; // edx
  _BOOL4 v12; // ecx
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // r11
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rdi
  int v29; // eax
  FILE *v30; // rsi
  __int64 v31; // [rsp+0h] [rbp-270h]
  __int64 v32; // [rsp+8h] [rbp-268h]
  int v33; // [rsp+8h] [rbp-268h]
  _BOOL4 v34; // [rsp+10h] [rbp-260h]
  __int64 v35; // [rsp+10h] [rbp-260h]
  __int64 v36; // [rsp+10h] [rbp-260h]
  __int64 v37; // [rsp+10h] [rbp-260h]
  __int64 v38; // [rsp+10h] [rbp-260h]
  __int64 v39; // [rsp+18h] [rbp-258h]
  int v40; // [rsp+24h] [rbp-24Ch] BYREF
  int v41; // [rsp+28h] [rbp-248h] BYREF
  char v42[4]; // [rsp+2Ch] [rbp-244h] BYREF
  __int64 v43; // [rsp+30h] [rbp-240h] BYREF
  __int64 v44; // [rsp+38h] [rbp-238h] BYREF
  char v45[160]; // [rsp+40h] [rbp-230h] BYREF
  _QWORD v46[2]; // [rsp+E0h] [rbp-190h] BYREF
  char v47; // [rsp+F1h] [rbp-17Fh]
  __int64 v48; // [rsp+138h] [rbp-138h]

  v2 = (_DWORD *)(a1 + 48);
  v4 = *(_QWORD *)(a1 + 288);
  v40 = 0;
  for ( i = v4; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v39 = sub_8D40F0(i);
  sub_6E2250(v45, &v43, 4, 1, a1, 0);
  v6 = a2;
  if ( *(_BYTE *)(a2 + 8) == 1 )
    v6 = *(_QWORD *)(a2 + 24);
  sub_6E6610(v6, v46, 0);
  if ( !(unsigned int)sub_8D23B0(v39) && !(unsigned int)sub_8D23E0(i) )
  {
    if ( !(unsigned int)sub_8D3A70(v39) || *(char *)(i + 168) < 0 )
    {
      v40 = 1;
      v37 = 0;
      goto LABEL_32;
    }
    v9 = v39;
    if ( *(_BYTE *)(v39 + 140) == 12 )
    {
      do
        v9 = *(_QWORD *)(v9 + 160);
      while ( *(_BYTE *)(v9 + 140) == 12 );
    }
    else
    {
      v9 = v39;
    }
    v10 = 0;
    v44 = 0;
    if ( (*(_BYTE *)(v46[0] + 140LL) & 0xFB) == 8 )
    {
      v38 = v9;
      v29 = sub_8D4C10(v46[0], dword_4F077C4 != 2);
      v9 = v38;
      v10 = v29;
    }
    v11 = 1;
    v12 = (*(_BYTE *)(a1 + 176) & 1) == 0;
    if ( v47 != 2 )
    {
      v31 = v9;
      v34 = (*(_BYTE *)(a1 + 176) & 1) == 0;
      v13 = sub_6ED0A0(v46);
      v9 = v31;
      v12 = v34;
      v11 = v13 != 0;
    }
    v35 = v9;
    v14 = sub_83DE00(v9, v10, v11, v12, (_DWORD)v2, (unsigned int)&v41, (__int64)v42, (__int64)&v44, (__int64)&v40);
    v15 = v35;
    if ( v40 )
    {
      sub_876E10(v35, v35, v2, 1, 0, 0);
      LODWORD(v18) = 0;
      v19 = v35;
      if ( dword_4D048B8 )
        goto LABEL_24;
      v37 = 0;
    }
    else
    {
      if ( v41 )
      {
        sub_685360(0x122u, v2, v35);
        goto LABEL_8;
      }
      if ( !v14 )
      {
        if ( v10 == 1 && !v44 )
        {
          sub_685360(0x14Cu, v2, v35);
        }
        else
        {
          v30 = (FILE *)sub_67DA80(0x14Eu, v2, v35);
          sub_87CA90(v44, v30);
          sub_685910((__int64)v30, v30);
        }
        goto LABEL_8;
      }
      v16 = v35;
      v36 = v14;
      v32 = v15;
      sub_8769C0(v14, (_DWORD)v2, v16, 0, 1, 1, 1, 0, 0);
      v17 = v36;
      v37 = 0;
      v18 = *(_QWORD *)(v17 + 88);
      if ( dword_4D048B8 )
      {
        v19 = v32;
LABEL_24:
        v33 = v18;
        v20 = sub_6EB2F0(v19, v19, v2, 0);
        LODWORD(v18) = v33;
        v37 = v20;
      }
    }
    if ( !v40 )
    {
      v21 = sub_6F5430(v18, 0, v4, 0, 1, 1, 0, 0, 1, 0, (__int64)v2);
      v7 = v21;
      if ( (unsigned int)sub_730800(v21) )
      {
        sub_6E6000(v21, 0, v22, v23, v24, v25);
        sub_6E5A30(v48, 4, 8);
        goto LABEL_9;
      }
      sub_6E5A30(v48, 4, 8);
      if ( !v40 )
        goto LABEL_28;
      goto LABEL_33;
    }
LABEL_32:
    v7 = sub_6EAFA0(7);
    sub_6E5A30(v48, 4, 8);
    if ( !v40 )
    {
LABEL_28:
      v26 = sub_6F6F40(v46, 0);
      *(_QWORD *)(v26 + 16) = *(_QWORD *)(v7 + 64);
      *(_QWORD *)(v7 + 64) = v26;
      *(_BYTE *)(v7 + 72) = *(_BYTE *)(v7 + 72) & 0xEE | 0x10;
      if ( v37 )
      {
        *(_QWORD *)(v7 + 16) = v37;
        *(_BYTE *)(v37 + 193) |= 0x40u;
        sub_734250(v7, 1);
        if ( (unsigned int)sub_8DBE70(i) )
        {
          sub_6E2920(v7);
          sub_6E2C70(v43, 1, a1, 0);
          goto LABEL_35;
        }
      }
      else if ( (unsigned int)sub_8DBE70(i) )
      {
        goto LABEL_9;
      }
      v27 = sub_8D4490(i);
      v28 = v7;
      v7 = sub_725A70(6);
      sub_63BA50(v28, v4, v39, v7, v27);
LABEL_34:
      sub_6E2920(v7);
      result = sub_6E2C70(v43, 1, a1, 0);
      if ( !v37 )
        goto LABEL_10;
LABEL_35:
      result = v37;
      *(_QWORD *)(v7 + 16) = v37;
      *(_BYTE *)(v37 + 193) |= 0x40u;
      goto LABEL_10;
    }
LABEL_33:
    *(_QWORD *)(v7 + 56) = sub_6F6F40(v46, 0);
    goto LABEL_34;
  }
  sub_685360(0xB1Eu, v2, v4);
  *(_QWORD *)(a1 + 288) = sub_72C930(2846);
LABEL_8:
  v7 = sub_6EAFA0(0);
  sub_6E5A30(v48, 4, 8);
LABEL_9:
  sub_6E2920(v7);
  result = sub_6E2C70(v43, 1, a1, 0);
LABEL_10:
  *(_QWORD *)(a1 + 144) = v7;
  return result;
}
