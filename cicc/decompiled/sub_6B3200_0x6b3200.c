// Function: sub_6B3200
// Address: 0x6b3200
//
__int64 __fastcall sub_6B3200(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rbx
  _BOOL4 v6; // r15d
  unsigned int v7; // r8d
  __int64 v8; // rsi
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned int v13; // r8d
  __int64 v14; // rsi
  bool v15; // zf
  char v16; // dl
  __int64 v17; // rax
  __int64 v18; // rax
  char i; // dl
  unsigned __int8 v20; // al
  __int64 v21; // rax
  char j; // dl
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rax
  unsigned int v32; // eax
  int v33; // eax
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rsi
  int v41; // r15d
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  unsigned __int8 v52; // al
  __int64 *v53; // rdx
  int v54; // eax
  int v55; // [rsp+8h] [rbp-328h]
  unsigned int v56; // [rsp+Ch] [rbp-324h]
  unsigned __int8 v57; // [rsp+10h] [rbp-320h]
  _BOOL4 v58; // [rsp+10h] [rbp-320h]
  __int64 v59; // [rsp+10h] [rbp-320h]
  unsigned __int16 v60; // [rsp+18h] [rbp-318h]
  __int64 *v61; // [rsp+18h] [rbp-318h]
  _BOOL4 v62; // [rsp+24h] [rbp-30Ch] BYREF
  unsigned int v63; // [rsp+28h] [rbp-308h] BYREF
  int v64; // [rsp+2Ch] [rbp-304h] BYREF
  __int64 v65; // [rsp+30h] [rbp-300h] BYREF
  __int64 v66; // [rsp+38h] [rbp-2F8h] BYREF
  _BYTE v67[352]; // [rsp+40h] [rbp-2F0h] BYREF
  __int64 v68[2]; // [rsp+1A0h] [rbp-190h] BYREF
  char v69; // [rsp+1B0h] [rbp-180h]
  char v70; // [rsp+1B1h] [rbp-17Fh]
  int v71; // [rsp+1E4h] [rbp-14Ch] BYREF
  __int64 v72; // [rsp+1ECh] [rbp-144h]
  __int64 v73; // [rsp+1F8h] [rbp-138h]

  v62 = 0;
  if ( a2 )
  {
    v5 = v67;
    v60 = *(_WORD *)(a2 + 8);
    sub_6F8AB0(a2, (unsigned int)v67, (unsigned int)v68, 0, (unsigned int)&v66, (unsigned int)&v63, 0);
  }
  else
  {
    v5 = a1;
    v60 = word_4F06418[0];
    v66 = *(_QWORD *)&dword_4F063F8;
    v63 = dword_4F06650[0];
    sub_7B8B50(a1, 0, a3, a4);
    sub_69ED20((__int64)v68, 0, 9, 0);
  }
  v57 = (v60 != 70) + 45;
  if ( (unsigned int)sub_68FE10(v5, 1, 1) || (unsigned int)sub_68FE10(v68, 0, 1) )
    sub_84EC30(v57, 0, 0, 1, 0, (_DWORD)v5, (__int64)v68, (__int64)&v66, v63, 0, 0, a3, 0, 0, (__int64)&v62);
  v6 = v62;
  if ( !v62 )
  {
    v64 = 0;
    if ( *((_BYTE *)v5 + 17) != 1 || (unsigned int)sub_6ED0A0(v5) || v70 != 1 || (unsigned int)sub_6ED0A0(v68) )
      goto LABEL_7;
    v31 = v68[0];
    if ( *v5 != v68[0] )
    {
      if ( !(unsigned int)sub_8D97D0(*v5, v68[0], 0, v29, v30) )
        goto LABEL_7;
      v31 = *v5;
    }
    if ( !(unsigned int)sub_8D3410(v31) )
    {
      v55 = 1;
      v7 = 4;
LABEL_8:
      v8 = v7;
      v56 = v7;
      sub_6F69D0(v5, v7);
      v9 = *v5;
      v10 = sub_8D2D80(*v5);
      v13 = v56;
      if ( !v10 )
      {
        v32 = sub_6E9530(v9, v8, v11, v12, v56);
        v33 = sub_6FB4D0(v5, v32);
        v13 = v56;
        v6 = v33 != 0;
      }
      sub_6F69D0(v68, v13);
      v14 = *v5;
      v15 = *((_BYTE *)v5 + 16) == 0;
      v65 = *v5;
      if ( v15 )
        goto LABEL_19;
      v16 = *(_BYTE *)(v14 + 140);
      if ( v16 == 12 )
      {
        v17 = v14;
        do
        {
          v17 = *(_QWORD *)(v17 + 160);
          v16 = *(_BYTE *)(v17 + 140);
        }
        while ( v16 == 12 );
      }
      if ( !v16 || !v69 )
        goto LABEL_19;
      v18 = v68[0];
      for ( i = *(_BYTE *)(v68[0] + 140); i == 12; i = *(_BYTE *)(v18 + 140) )
        v18 = *(_QWORD *)(v18 + 160);
      if ( !i )
      {
LABEL_19:
        v65 = sub_72C930(v68);
        v20 = sub_6E9930(v60, v65);
        if ( v55 )
          goto LABEL_20;
        goto LABEL_47;
      }
      if ( v55 )
      {
        v20 = sub_6E9930(v60, v14);
LABEL_20:
        sub_6F7B90(v5, v68, v20, v65, 1, a3);
        if ( *(_BYTE *)(a3 + 16) )
        {
          v21 = *(_QWORD *)a3;
          for ( j = *(_BYTE *)(*(_QWORD *)a3 + 140LL); j == 12; j = *(_BYTE *)(v21 + 140) )
            v21 = *(_QWORD *)(v21 + 160);
          if ( j )
          {
            v23 = v5[11];
            if ( v23 )
            {
              if ( v73 )
              {
                v24 = v5[11];
                do
                {
                  v25 = v24;
                  v24 = *(_QWORD *)(v24 + 48);
                }
                while ( v24 );
                *(_QWORD *)(v25 + 48) = v73;
              }
            }
            else
            {
              v23 = v73;
            }
            *(_QWORD *)(a3 + 88) = v23;
          }
        }
        goto LABEL_30;
      }
      v34 = v14;
      if ( (unsigned int)sub_8D2AF0(v14) )
      {
        if ( (unsigned int)sub_6E5430(v14, v14, v35, v36, v37, v38) )
        {
          v34 = 1044;
          sub_6851C0(0x414u, (_DWORD *)v5 + 17);
        }
      }
      else
      {
        v34 = v68[0];
        if ( !(unsigned int)sub_8D2AF0(v68[0]) )
        {
          if ( !v6 && !(unsigned int)sub_8D2E30(v68[0]) )
          {
            if ( (unsigned int)sub_6E9580(v68) )
            {
              v58 = sub_68B1F0(v5, v68, &v64);
              v65 = sub_6E8B10(v5, v68, v49, v50, v51);
              v52 = sub_6E9930(v60, v65);
              v41 = v52;
              sub_6FC7D0(v65, v5, v68, v52);
              if ( v58 )
              {
                v53 = v5;
                if ( v64 )
                  v53 = v68;
                if ( *((_BYTE *)v53 + 16) == 2 )
                {
                  v61 = v53;
                  v59 = (__int64)(v53 + 18);
                  if ( (unsigned int)sub_8D2930(v53[34]) )
                  {
                    if ( *((_BYTE *)v61 + 317) == 1 )
                    {
                      v54 = sub_6210B0(v59, 0);
                      if ( v54 )
                      {
                        if ( v54 < 0 && (unsigned int)sub_6E53E0(5, 514, &v66) )
                          sub_684B30(0x202u, &v66);
                      }
                      else if ( (unsigned int)sub_6E53E0(5, 186, &v66) )
                      {
                        sub_684B30(0xBAu, &v66);
                      }
                    }
                  }
                }
              }
              goto LABEL_48;
            }
            v65 = sub_6E8B10(v5, v68, v46, v47, v48);
            v40 = v65;
            goto LABEL_46;
          }
          sub_6EB6C0((_DWORD)v5, (unsigned int)v68, (unsigned int)&v66, v57, 0, 0, 0, 0, (__int64)&v65);
          v39 = v65;
LABEL_45:
          v40 = v39;
LABEL_46:
          v20 = sub_6E9930(v60, v40);
LABEL_47:
          v41 = v20;
          sub_6FC7D0(v65, v5, v68, v20);
LABEL_48:
          sub_7016A0(v41, (_DWORD)v5, (unsigned int)v68, v65, a3, (unsigned int)&v66, v63);
          goto LABEL_30;
        }
        if ( (unsigned int)sub_6E5430(v34, v14, v42, v43, v44, v45) )
        {
          v34 = 1044;
          sub_6851C0(0x414u, &v71);
        }
      }
      v39 = sub_72C930(v34);
      v65 = v39;
      goto LABEL_45;
    }
LABEL_7:
    v55 = 0;
    v7 = 0;
    goto LABEL_8;
  }
LABEL_30:
  v26 = *((_DWORD *)v5 + 17);
  *(_WORD *)(a3 + 72) = *((_WORD *)v5 + 36);
  *(_DWORD *)(a3 + 68) = v26;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
  v27 = v72;
  *(_QWORD *)(a3 + 76) = v72;
  unk_4F061D8 = v27;
  sub_6E3280(a3, &v66);
  return sub_6E3BA0(a3, &v66, v63, 0);
}
