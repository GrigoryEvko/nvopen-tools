// Function: sub_6BFEC0
// Address: 0x6bfec0
//
__int64 __fastcall sub_6BFEC0(__int64 a1, __int64 *a2, _DWORD *a3, _DWORD *a4, _BYTE *a5)
{
  __int64 v8; // r12
  char v9; // al
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 result; // rax
  _BYTE *v14; // r14
  char v15; // al
  __int64 v16; // r11
  int v17; // eax
  char v18; // r14
  int v19; // r8d
  __int64 v20; // r14
  __int64 v21; // rax
  int v22; // r9d
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // r8
  char v27; // cl
  char i; // dl
  char v29; // al
  int v30; // eax
  _BOOL8 v31; // rsi
  unsigned int v32; // r14d
  __int64 v33; // r11
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdi
  __int64 v47; // [rsp+0h] [rbp-90h]
  __int64 v48; // [rsp+0h] [rbp-90h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  __int64 v50; // [rsp+8h] [rbp-88h]
  __int64 v51; // [rsp+8h] [rbp-88h]
  __int64 v52; // [rsp+8h] [rbp-88h]
  __int64 v53; // [rsp+8h] [rbp-88h]
  int v54; // [rsp+10h] [rbp-80h]
  int v55; // [rsp+10h] [rbp-80h]
  int v57; // [rsp+24h] [rbp-6Ch]
  int v58; // [rsp+28h] [rbp-68h]
  __int64 v59; // [rsp+28h] [rbp-68h]
  int v60; // [rsp+28h] [rbp-68h]
  int v61; // [rsp+30h] [rbp-60h] BYREF
  int v62; // [rsp+34h] [rbp-5Ch] BYREF
  int v63; // [rsp+38h] [rbp-58h] BYREF
  int v64; // [rsp+3Ch] [rbp-54h] BYREF
  unsigned int v65; // [rsp+40h] [rbp-50h] BYREF
  int v66; // [rsp+44h] [rbp-4Ch] BYREF
  __int64 v67; // [rsp+48h] [rbp-48h] BYREF
  __int64 v68; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v69[7]; // [rsp+58h] [rbp-38h] BYREF

  v8 = a1;
  v61 = 0;
  v62 = 0;
  v57 = sub_8D32E0(a1);
  v58 = sub_8D2600(a1);
  v9 = *((_BYTE *)a2 + 16);
  if ( v9 == 1 )
  {
    v10 = a2[18];
  }
  else
  {
    v10 = 0;
    if ( v9 == 2 )
    {
      v10 = a2[36];
      if ( !v10 && *((_BYTE *)a2 + 317) == 12 && *((_BYTE *)a2 + 320) == 1 )
        v10 = sub_72E9A0(a2 + 18);
    }
  }
  sub_6BEBB0(a1, a2, 3u, &v64, &v63, &v61);
  if ( v63 )
  {
    v62 = 1;
    if ( !v61 )
      goto LABEL_5;
    return sub_6E6840(a2);
  }
  if ( unk_4D041FC )
  {
    v23 = *a2;
    if ( a1 == *a2 || (unsigned int)sub_8D97D0(a1, v23, 0, v11, v12) )
    {
      result = sub_8D3410(a1);
      if ( (_DWORD)result || *((_BYTE *)a2 + 17) != 2 )
      {
        v62 = 1;
        if ( !v61 )
          return result;
        return sub_6E6840(a2);
      }
    }
  }
  if ( v62 )
    goto LABEL_49;
  v54 = sub_68B3F0(a2, a1);
  if ( !((v54 != 0) | v58 | v57) )
  {
    if ( !dword_4F077BC )
      goto LABEL_18;
    if ( (_DWORD)qword_4F077B4 )
      goto LABEL_18;
    if ( (unsigned __int64)(qword_4F077A8 - 30400LL) > 0x27D7 )
      goto LABEL_18;
    if ( !(unsigned int)sub_8D2E30(a1) )
      goto LABEL_18;
    if ( !(unsigned int)sub_8D2E30(*a2) )
      goto LABEL_18;
    v50 = sub_8D46C0(a1);
    v47 = sub_8D46C0(*a2);
    if ( (*(_BYTE *)(v50 + 140) & 0xFB) != 8 || !(unsigned int)sub_8D4C10(v50, dword_4F077C4 != 2) )
      goto LABEL_18;
    v25 = v47;
    v26 = v50;
    v27 = *(_BYTE *)(v47 + 140);
    i = v27;
    if ( (v27 & 0xFB) == 8 )
    {
      v46 = v47;
      v48 = v50;
      v53 = v25;
      if ( (unsigned int)sub_8D4C10(v46, dword_4F077C4 != 2) )
        goto LABEL_18;
      v26 = v48;
      v25 = v53;
      v29 = *(_BYTE *)(v48 + 140);
      v27 = *(_BYTE *)(v53 + 140);
      if ( v29 != 12 )
        goto LABEL_60;
    }
    else
    {
      v29 = *(_BYTE *)(v50 + 140);
      if ( v29 != 12 )
      {
LABEL_62:
        if ( (unsigned __int8)(v29 - 9) <= 2u && (unsigned __int8)(i - 9) <= 2u )
        {
          v51 = v26;
          if ( sub_8D5CE0(v26, v25) )
            v8 = sub_72D2E0(v51, 0);
        }
LABEL_18:
        sub_6F69D0(a2, 8);
        sub_6F6890(a2, 0);
        if ( (unsigned int)sub_69A8F0(a2, v8, 0, a4, a5) )
        {
          v16 = *a2;
          v17 = v61;
          v67 = v8;
          v68 = v16;
          goto LABEL_20;
        }
LABEL_40:
        v24 = *a2;
        v61 = 1;
        v67 = v8;
        v68 = v24;
        return sub_6E6840(a2);
      }
    }
    do
    {
      v26 = *(_QWORD *)(v26 + 160);
      v29 = *(_BYTE *)(v26 + 140);
    }
    while ( v29 == 12 );
LABEL_60:
    for ( i = v27; i == 12; i = *(_BYTE *)(v25 + 140) )
      v25 = *(_QWORD *)(v25 + 160);
    goto LABEL_62;
  }
  if ( (unsigned int)sub_8D3110(a1) && (unsigned int)sub_6ECD10(a2) )
    sub_6F69D0(a2, 0);
  if ( !(unsigned int)sub_69A8F0(a2, a1, 0, a4, a5) )
    goto LABEL_40;
  v16 = *a2;
  v67 = a1;
  v17 = v61;
  v68 = v16;
  if ( v57 )
  {
    v49 = v16;
    if ( v61 )
      return sub_6E6840(a2);
    sub_68F0D0(a1, a2, v64, 0, 3u, a4, &v67, &v68, &v62);
    v17 = v61;
    v16 = v49;
  }
LABEL_20:
  if ( v17 )
    return sub_6E6840(a2);
  if ( !v62 )
  {
    v18 = *((_BYTE *)a2 + 16);
    v19 = (_DWORD)a2 + 144;
    if ( v18 != 2 )
    {
      v19 = 0;
      if ( v18 == 3 )
      {
        sub_6FC070(v8, a2, 1, 1, 0);
        if ( v61 )
          return sub_6E6840(a2);
        goto LABEL_5;
      }
    }
    if ( v58 )
    {
      sub_6F7220(a2, v8);
      if ( v61 )
        return sub_6E6840(a2);
      goto LABEL_5;
    }
    if ( v54 )
    {
      v20 = a2[11];
      v21 = sub_6F6F40(a2, 0);
      v59 = sub_691700(v21, v8, 0);
      sub_6E2DD0(a2, 1);
      v22 = v61;
      *a2 = v8;
      a2[11] = v20;
      a2[18] = v59;
      if ( v22 )
        return sub_6E6840(a2);
      goto LABEL_5;
    }
    v52 = v16;
    v55 = v19;
    v60 = v67;
    v30 = sub_6EB660(a2);
    v31 = v18 == 2;
    if ( (unsigned int)sub_8E2F00(v68, v31, (*((_BYTE *)a2 + 19) & 0x10) != 0, v30, v55, v60, 0, 171, (__int64)&v65) )
    {
      v32 = v65;
      v66 = 0;
      v33 = v52;
      if ( v65 )
      {
        v39 = sub_6E53E0(5, v65, a3);
        v33 = v52;
        if ( v39 )
        {
          sub_684B30(v32, a3);
          v33 = v52;
        }
      }
      if ( (dword_4F04C44 != -1
         || (v34 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v34 + 6) & 6) != 0)
         || *(_BYTE *)(v34 + 4) == 12)
        && ((unsigned int)sub_8DBE70(v33) || (unsigned int)sub_8DBE70(v8)) )
      {
        sub_6F4200(a2, v8, 3, 0);
      }
      else
      {
        if ( (unsigned int)sub_6E6010()
          && (unsigned int)sub_8D3D10(v68)
          && (unsigned int)sub_8D3D10(v67)
          && (unsigned int)sub_8D5F90(v68, v67, &v66, v69)
          && v66
          && (*(_BYTE *)(v69[0] + 96LL) & 4) == 0
          && !(unsigned int)sub_87DF20(v69[0]) )
        {
          sub_6E5D20(7, 269, a3, *(_QWORD *)(v69[0] + 40LL));
        }
        else if ( (unsigned int)sub_6E6010()
               && dword_4F077C4 == 2
               && (unsigned int)sub_8D2F30(v68, v67)
               && (unsigned int)sub_8D5EF0(v68, v67, &v66, v69)
               && !v66
               && (*(_BYTE *)(v69[0] + 96LL) & 4) == 0
               && !(unsigned int)sub_87DF20(v69[0])
               && (!dword_4F077BC || qword_4F077A8 > 0x76BFu) )
        {
          sub_6E5D20(7, 1280, a3, *(_QWORD *)(v69[0] + 40LL));
        }
        if ( v57 )
          sub_6FAB30(a2, v8, 1, 0, 0);
        else
          sub_6FCCE0(v8, (_DWORD)a2, (_DWORD)a4, 1, 0, 0, 0);
      }
    }
    else
    {
      v61 = 1;
      if ( (unsigned int)sub_8D3A70(v8) )
      {
        if ( (unsigned int)sub_6E5430(v8, v31, v35, v36, v37, v38) )
        {
          sub_685360(0x77u, a4, v8);
          if ( v61 )
            return sub_6E6840(a2);
          goto LABEL_5;
        }
      }
      else
      {
        v40 = v67;
        v41 = v68;
        if ( (unsigned int)sub_8DEFB0(v68, v67, 1, 0)
          && (v40 = v67, v41 = v68, (unsigned int)sub_8DF7D0(v68, v67, &v65)) )
        {
          if ( (unsigned int)sub_6E5430(v41, v40, v42, v43, v44, v45) )
          {
            sub_6851A0(0x2B6u, a3, (__int64)"static_cast");
            if ( v61 )
              return sub_6E6840(a2);
            goto LABEL_5;
          }
        }
        else if ( (unsigned int)sub_6E5430(v41, v40, v42, v43, v44, v45) )
        {
          sub_6851C0(0xABu, a3);
          if ( v61 )
            return sub_6E6840(a2);
          goto LABEL_5;
        }
      }
    }
LABEL_49:
    if ( v61 )
      return sub_6E6840(a2);
  }
LABEL_5:
  result = sub_6E3FE0(v10, 3, a2);
  v14 = (_BYTE *)result;
  if ( result )
  {
    if ( (unsigned int)sub_730740(result) || (v15 = v14[24], v15 == 5) || v15 == 1 && (v14[59] & 1) != 0 )
      v14[25] |= 0x40u;
    return sub_6E3CB0(v14, a3, a4, v8);
  }
  return result;
}
