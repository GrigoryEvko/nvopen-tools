// Function: sub_8E2840
// Address: 0x8e2840
//
_BOOL8 __fastcall sub_8E2840(
        __int64 a1,
        int a2,
        int a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        _DWORD *a9,
        _DWORD *a10)
{
  __int64 v11; // r12
  int v12; // r15d
  int v13; // r14d
  char i; // al
  _BOOL8 result; // rax
  int v16; // r10d
  int v17; // edx
  char v18; // al
  int v19; // eax
  int v20; // r10d
  _BOOL4 v21; // eax
  int v22; // eax
  _BOOL4 v23; // eax
  _BOOL4 v24; // eax
  _BOOL4 v25; // eax
  char v26; // al
  _BOOL4 v27; // eax
  __int64 v28; // rax
  __int64 v29; // r11
  int v30; // r12d
  int v31; // eax
  _BOOL4 v32; // eax
  _BOOL4 v33; // eax
  int v34; // eax
  int v35; // eax
  _BOOL4 v36; // eax
  int v37; // eax
  int v38; // eax
  _BOOL4 v39; // eax
  int v40; // eax
  int v41; // eax
  int v42; // eax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // rdx
  char v48; // al
  _BOOL4 v49; // eax
  __int64 v50; // [rsp+0h] [rbp-A0h]
  int v51; // [rsp+8h] [rbp-98h]
  int v52; // [rsp+8h] [rbp-98h]
  int v53; // [rsp+10h] [rbp-90h]
  __int64 v54; // [rsp+10h] [rbp-90h]
  int v55; // [rsp+18h] [rbp-88h]
  int v56; // [rsp+18h] [rbp-88h]
  int v57; // [rsp+18h] [rbp-88h]
  int v58; // [rsp+18h] [rbp-88h]
  __int64 v59; // [rsp+18h] [rbp-88h]
  int v60; // [rsp+18h] [rbp-88h]
  int v61; // [rsp+18h] [rbp-88h]
  int v62; // [rsp+18h] [rbp-88h]
  int v63; // [rsp+18h] [rbp-88h]
  int v64; // [rsp+20h] [rbp-80h] BYREF
  int v65; // [rsp+24h] [rbp-7Ch] BYREF
  __int64 v66; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v67[8]; // [rsp+30h] [rbp-70h] BYREF
  int v68; // [rsp+38h] [rbp-68h]
  char v69; // [rsp+3Dh] [rbp-63h]
  __int128 v70; // [rsp+50h] [rbp-50h] BYREF
  __int64 v71; // [rsp+60h] [rbp-40h]

  v11 = a6;
  v12 = unk_4D04200;
  unk_4D04200 = 0;
  *a9 = 0;
  *a10 = 0;
  v13 = dword_4D04964;
  if ( dword_4D04964 )
    v13 = byte_4F07472[0] == 8;
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  for ( i = *(_BYTE *)(a6 + 140); i == 12; i = *(_BYTE *)(v11 + 140) )
    v11 = *(_QWORD *)(v11 + 160);
  if ( i == 1 )
    goto LABEL_16;
  if ( (*(_BYTE *)(v11 + 141) & 0x20) == 0 )
  {
    v16 = sub_8E1010(a1, a2, a3, a4, 0, a5, v11, 0, a7, v13, a8, (__int64)v67, 16778248);
    if ( v16 )
    {
      v17 = v68;
      v18 = v69;
      if ( !v68 || v68 == 1713 || (v69 & 0x10) != 0 || v68 == 1419 )
      {
        result = (v69 & 0x10) != 0;
        *a9 = v68;
        *a10 = result;
        if ( result && *a9 == 188 )
        {
          *a9 = 0;
          *a10 = 0;
          goto LABEL_17;
        }
        goto LABEL_16;
      }
      if ( dword_4F077C4 != 2 )
        goto LABEL_15;
    }
    else if ( dword_4F077C4 != 2 )
    {
      goto LABEL_9;
    }
    v55 = v16;
    v71 = 0;
    v70 = 0;
    v19 = sub_8D2F30(a1, v11);
    v20 = v55;
    if ( v19 && (v27 = sub_8D5EF0(a1, v11, &v64, &v66), v20 = v55, v27) && !v64 )
    {
      if ( (*(_BYTE *)(v66 + 96) & 2) == 0 )
      {
        v53 = v55;
        v59 = sub_8D46C0(a1);
        v28 = sub_8D46C0(v11);
        v20 = v53;
        v29 = v28;
        goto LABEL_40;
      }
    }
    else
    {
      v56 = v20;
      v21 = sub_8D3D10(a1);
      v20 = v56;
      if ( v21 )
      {
        v32 = sub_8D3D10(v11);
        v20 = v56;
        if ( v32 )
        {
          v33 = sub_8D5F90(a1, v11, &v64, &v66);
          v20 = v56;
          if ( v33 )
          {
            if ( v64 )
            {
              v51 = v56;
              v59 = sub_8D4870(a1);
              v54 = sub_8D4870(v11);
              v34 = sub_8E0BF0(v54, v59, 0, 0, &v65);
              v29 = v54;
              v20 = v51;
              if ( v34 )
              {
LABEL_40:
                if ( (*(_BYTE *)(v59 + 140) & 0xFB) == 8 )
                {
                  v50 = v29;
                  v52 = v20;
                  v40 = sub_8D4C10(v59, dword_4F077C4 != 2);
                  v29 = v50;
                  v30 = v40;
                  v20 = v52;
                  if ( (*(_BYTE *)(v50 + 140) & 0xFB) != 8 )
                  {
                    v31 = 0;
                    goto LABEL_44;
                  }
                }
                else
                {
                  if ( (*(_BYTE *)(v29 + 140) & 0xFB) != 8 )
                    goto LABEL_29;
                  v30 = 0;
                }
                v60 = v20;
                v31 = sub_8D4C10(v29, dword_4F077C4 != 2);
                v20 = v60;
LABEL_44:
                if ( v31 != v30 && (v30 & ~v31) != 0 )
                  goto LABEL_46;
                goto LABEL_29;
              }
            }
          }
        }
      }
    }
    v57 = v20;
    v22 = sub_8D29A0(a1);
    v20 = v57;
    if ( !v22 )
    {
      v23 = sub_8D2660(v11);
      v20 = v57;
      if ( !v23 )
      {
        v41 = sub_8E1010(v11, 0, 0, 0, 0, 0, a1, 0, a7, v13, 171, (__int64)&v70, 16778248);
        v20 = v57;
        if ( v41 )
          goto LABEL_25;
      }
    }
    v58 = v20;
    v24 = sub_8D2710(a1);
    v20 = v58;
    if ( v24 && (v25 = sub_8D2EB0(v11), v20 = v58, v25) )
    {
LABEL_25:
      if ( a7 )
        goto LABEL_29;
      v26 = *(_BYTE *)(a1 + 140);
      if ( v26 == 6 )
      {
        if ( (*(_BYTE *)(a1 + 168) & 1) != 0 || *(_BYTE *)(v11 + 140) != 6 || (*(_BYTE *)(v11 + 168) & 1) != 0 )
          goto LABEL_29;
      }
      else if ( v26 != 13 || *(_BYTE *)(v11 + 140) != 13 )
      {
        goto LABEL_29;
      }
      v63 = v20;
      v42 = sub_8DF7D0(a1, v11, 0);
      v20 = v63;
      if ( !v42 )
      {
        if ( !dword_4F06978 )
          goto LABEL_29;
        if ( *(_BYTE *)(a1 + 140) == 13 )
        {
          v45 = *(_QWORD *)(a1 + 168);
          v46 = *(_QWORD *)(v11 + 168);
        }
        else
        {
          v45 = *(_QWORD *)(a1 + 160);
          v46 = *(_QWORD *)(v11 + 160);
        }
        while ( 1 )
        {
          v47 = *(unsigned __int8 *)(v45 + 140);
          if ( (_BYTE)v47 != 12 )
            break;
          v45 = *(_QWORD *)(v45 + 160);
        }
        while ( 1 )
        {
          v48 = *(_BYTE *)(v46 + 140);
          if ( v48 != 12 )
            break;
          v46 = *(_QWORD *)(v46 + 160);
        }
        if ( (_BYTE)v47 != 7 || v48 != 7 || (v49 = sub_8DADD0(v45, v46, v47, v43, v44), v20 = v63, !v49) )
        {
LABEL_29:
          v17 = DWORD2(v70);
          v18 = BYTE13(v70);
          if ( !DWORD2(v70) || (BYTE13(v70) & 0x10) != 0 || !v20 )
            goto LABEL_15;
          goto LABEL_32;
        }
      }
    }
    else
    {
      v61 = v20;
      v35 = sub_8D2870(v11);
      v20 = v61;
      if ( v35 )
      {
        if ( *(_BYTE *)(a1 + 140) == 2 )
          goto LABEL_29;
        v36 = sub_8D2A90(a1);
        v20 = v61;
        if ( v36 )
          goto LABEL_29;
      }
      v62 = v20;
      v37 = sub_8D2870(a1);
      v20 = v62;
      if ( v37 )
      {
        v38 = sub_8D2780(v11);
        v20 = v62;
        if ( v38 )
          goto LABEL_29;
        v39 = sub_8D2A90(v11);
        v20 = v62;
        if ( v39 )
          goto LABEL_29;
      }
    }
LABEL_46:
    if ( !v20 )
    {
      result = 0;
      goto LABEL_17;
    }
LABEL_32:
    v17 = v68;
    v18 = v69;
LABEL_15:
    *a9 = v17;
    *a10 = (v18 & 0x10) != 0;
LABEL_16:
    result = 1;
    goto LABEL_17;
  }
LABEL_9:
  result = 0;
LABEL_17:
  unk_4D04200 = v12;
  return result;
}
