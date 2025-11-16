// Function: sub_6D53F0
// Address: 0x6d53f0
//
__int64 __fastcall sub_6D53F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  char v11; // r14
  _BOOL4 v12; // r14d
  __int64 v13; // rdx
  __int64 v14; // rcx
  _BOOL4 v15; // r15d
  __int64 v16; // rax
  char *IO_save_base; // rbx
  char v18; // dl
  char *v19; // rax
  __int64 v20; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  char *v29; // r15
  char v30; // dl
  __int64 v31; // rbx
  __int64 v32; // r14
  char v33; // al
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned int v36; // r8d
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // rdi
  char v42; // al
  __int64 v43; // r8
  __int64 v44; // rdi
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r10
  __int64 v50; // rax
  unsigned int v51; // r15d
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // [rsp+0h] [rbp-240h]
  char v55; // [rsp+8h] [rbp-238h]
  __int64 v56; // [rsp+10h] [rbp-230h]
  __int64 v57; // [rsp+10h] [rbp-230h]
  __int64 v58; // [rsp+10h] [rbp-230h]
  unsigned int v59; // [rsp+10h] [rbp-230h]
  _QWORD *v60; // [rsp+18h] [rbp-228h]
  int v61; // [rsp+24h] [rbp-21Ch]
  int v62; // [rsp+24h] [rbp-21Ch]
  _BOOL4 v63; // [rsp+28h] [rbp-218h]
  __int64 v64; // [rsp+28h] [rbp-218h]
  int v65; // [rsp+34h] [rbp-20Ch] BYREF
  int v66; // [rsp+38h] [rbp-208h] BYREF
  __int64 v67; // [rsp+3Ch] [rbp-204h] BYREF
  int v68; // [rsp+44h] [rbp-1FCh] BYREF
  __int64 v69; // [rsp+48h] [rbp-1F8h] BYREF
  __int64 v70; // [rsp+50h] [rbp-1F0h] BYREF
  char *v71; // [rsp+58h] [rbp-1E8h] BYREF
  __int64 v72; // [rsp+60h] [rbp-1E0h] BYREF
  FILE v73[2]; // [rsp+68h] [rbp-1D8h] BYREF

  v6 = a2;
  v7 = a1;
  v66 = 0;
  HIDWORD(v67) = 0;
  v69 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( a1 )
  {
    sub_6E43A0(a1, &v71, &v72, 0, 0);
    a2 = (__int64 *)a1;
    v11 = *v71;
    a1 = *((_QWORD *)v71 + 3);
    *(_QWORD *)&v73[0]._flags = v72;
    v12 = (v11 & 0x10) != 0;
    v63 = (*v71 & 8) != 0;
    sub_6F8800(a1, v7, &v73[0]._IO_save_base);
  }
  else
  {
    v12 = 0;
    v63 = 0;
    v72 = *(_QWORD *)&dword_4F063F8;
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
  {
    a1 = (__int64)&v72;
    v15 = sub_6E9250(&v72) != 0;
    if ( v7 )
      goto LABEL_7;
    goto LABEL_21;
  }
  v15 = 1;
  if ( (unsigned int)sub_6E5430(a1, a2, v8, v9, v10) )
  {
    a2 = &v72;
    a1 = 57;
    sub_6851C0(0x39u, &v72);
  }
  if ( !v7 )
  {
LABEL_21:
    if ( word_4F06418[0] == 146 )
    {
      sub_7B8B50(a1, a2, v13, v14);
      v12 = 1;
    }
    *(_QWORD *)&v73[0]._flags = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(a1, a2, v13, v14);
    v63 = 0;
    if ( word_4F06418[0] == 25 )
    {
      sub_7B8B50(a1, a2, v22, v23);
      ++*(_BYTE *)(qword_4F061C8 + 34LL);
      ++*(_QWORD *)(qword_4D03C50 + 40LL);
      if ( word_4F06418[0] != 26 )
      {
        sub_6E5C80(unk_4F07470, 387, &dword_4F063F8);
        sub_6D4F40(1, 0, 0, &v65, &v70, v69);
      }
      sub_7BE280(26, 17, 0, 0);
      v63 = 1;
      --*(_BYTE *)(qword_4F061C8 + 34LL);
      --*(_QWORD *)(qword_4D03C50 + 40LL);
    }
    sub_69ED20((__int64)&v73[0]._IO_save_base, 0, 18, 0);
  }
LABEL_7:
  if ( (dword_4F04C44 != -1
     || (v16 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v16 + 6) & 6) != 0)
     || *(_BYTE *)(v16 + 4) == 12)
    && ((unsigned int)sub_8DD3B0(v73[0]._IO_save_base)
     || (unsigned int)sub_8D2EF0(v73[0]._IO_save_base)
     && (v35 = sub_8D46C0(v73[0]._IO_save_base), (unsigned int)sub_8DD3B0(v35))) )
  {
    v61 = 1;
  }
  else
  {
    v61 = sub_8D3A70(v73[0]._IO_save_base);
    if ( v61 )
    {
      v34 = 4;
      if ( dword_4F077C4 == 2 )
      {
        v34 = 8;
        if ( unk_4F07778 <= 201102 )
          v34 = dword_4F07774 == 0 ? 4 : 8;
      }
      sub_845C60(&v73[0]._IO_save_base, 0, v34, 0, &v66);
      v61 = 0;
      if ( !v66 )
        goto LABEL_38;
      goto LABEL_12;
    }
  }
  if ( !v66 )
LABEL_38:
    sub_6F69D0(&v73[0]._IO_save_base, 0);
LABEL_12:
  if ( LOBYTE(v73[0]._IO_save_end) )
  {
    IO_save_base = v73[0]._IO_save_base;
    v18 = v73[0]._IO_save_base[140];
    if ( v18 == 12 )
    {
      v19 = v73[0]._IO_save_base;
      do
      {
        v19 = (char *)*((_QWORD *)v19 + 20);
        v18 = v19[140];
      }
      while ( v18 == 12 );
    }
    if ( v18 )
    {
      if ( v15 )
        goto LABEL_18;
      if ( v61 )
      {
        v24 = *(_QWORD *)&dword_4D03B80;
        v55 = v63;
        goto LABEL_28;
      }
      if ( (unsigned int)sub_8D2E30(v73[0]._IO_save_base) )
      {
        v24 = sub_8D46C0(IO_save_base);
        if ( (unsigned int)sub_8D2310(v24) )
        {
          sub_6E68E0(435, &v73[0]._IO_save_base);
          goto LABEL_18;
        }
        if ( !v63 && (unsigned int)sub_8D3410(v24) )
        {
          v51 = 1765 - ((dword_4D04964 == 0) - 1);
          if ( (unsigned int)sub_6E53E0(5, v51, (char *)&v73[0]._lock + 4) )
            sub_684B30(v51, (_DWORD *)&v73[0]._lock + 1);
          v52 = sub_8D67C0(v24);
          sub_6FC3F0(v52, &v73[0]._IO_save_base, 1);
          v55 = 1;
          v63 = 1;
          v24 = sub_8D46C0(v73[0]._IO_save_base);
          goto LABEL_28;
        }
        if ( !(unsigned int)sub_8D2600(v24) )
        {
          v55 = v63;
          if ( (unsigned int)sub_8D2BE0(v24) )
            sub_685360(0xD56u, (_DWORD *)&v73[0]._lock + 1, v24);
LABEL_28:
          v56 = sub_6F6F40(&v73[0]._IO_save_base, 0);
          v60 = (_QWORD *)sub_726700(7);
          *v60 = sub_72CBE0(7, 0, v25, v26, v27, v28);
          v29 = (char *)v60[7];
          v30 = *v29;
          *((_QWORD *)v29 + 1) = v24;
          *((_QWORD *)v29 + 3) = v56;
          *v29 = v30 & 0xE6 | (8 * v55) | (16 * v12);
          v31 = sub_8D4130(v24);
          if ( (unsigned int)sub_8D3A70(v31) && dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v31) )
            sub_8AE000(v31);
          if ( v61 )
          {
            if ( !(unsigned int)sub_8D3A70(v31) )
            {
LABEL_32:
              v32 = 0;
LABEL_33:
              v33 = BYTE4(v67);
              *((_QWORD *)v29 + 2) = v32;
              *v29 = (4 * (v33 & 1)) | *v29 & 0xFB;
              sub_6E3AC0(v60, &v72, 0, 0);
              sub_6E70E0(v60, v6);
              goto LABEL_19;
            }
            v32 = 0;
            goto LABEL_72;
          }
          if ( unk_4D04844 )
          {
            v36 = !v63 ? 2 : 4;
            if ( v12 )
              goto LABEL_45;
          }
          else
          {
            v36 = 2;
            if ( v63 || v12 )
            {
LABEL_45:
              v32 = sub_7D3810(v36);
              if ( !v32 )
                goto LABEL_55;
LABEL_46:
              v37 = sub_87C270(v32, v31, &v68);
              v41 = v37;
              if ( v68 )
              {
                if ( (unsigned int)sub_6E5430(v37, v31, v38, v39, v40) )
                  sub_6854C0(0x10Au, v73, v32);
              }
              else
              {
                if ( v37 )
                {
                  v42 = *(_BYTE *)(v37 + 80);
                  v43 = v41;
                  if ( v42 == 16 )
                  {
                    v43 = **(_QWORD **)(v41 + 88);
                    v42 = *(_BYTE *)(v43 + 80);
                  }
                  if ( v42 == 24 )
                    v43 = *(_QWORD *)(v43 + 88);
                  v57 = *(_QWORD *)(v43 + 88);
                  if ( (*(_BYTE *)(v41 + 81) & 0x10) != 0 )
                  {
                    v54 = v43;
                    sub_878710(v41, &v73[0]._IO_read_ptr);
                    v73[0]._IO_read_end = *(char **)&v73[0]._flags;
                    sub_6E6370(&v73[0]._IO_read_ptr, v32);
                    v43 = v54;
                  }
                  sub_8767A0(4, v43, v73, 0);
                  v32 = v57;
LABEL_55:
                  if ( !(unsigned int)sub_8D3A70(v31) )
                  {
                    if ( !v63 )
                      goto LABEL_65;
                    v58 = 0;
                    goto LABEL_58;
                  }
LABEL_72:
                  if ( (unsigned int)sub_8D23B0(v31) && (unsigned int)sub_6E53E0(5, 414, (char *)&v73[0]._lock + 4) )
                    sub_684B30(0x19Eu, (_DWORD *)&v73[0]._lock + 1);
                  v58 = sub_6EB2F0(v31, v31, (char *)&v73[0]._lock + 4, 1);
                  v62 = v63 & (v61 ^ 1);
                  if ( v58 )
                  {
                    if ( !v32 || !unk_4D04814 || !(unsigned int)sub_87ADD0(v32, &v67, (char *)&v67 + 4, &v68) || !v68 )
                    {
                      v48 = sub_6EAFA0(0);
                      v49 = v48;
                      if ( v63 )
                      {
                        v64 = v48;
                        v53 = sub_7259C0(8);
                        *(_QWORD *)(v53 + 160) = v31;
                        v49 = sub_6EAFD0(v64, v53, v31, v58, 0);
                      }
                      v50 = qword_4D03C50;
                      *(_QWORD *)(v49 + 16) = v58;
                      if ( (*(_BYTE *)(v50 + 17) & 4) != 0 )
                        *(_BYTE *)(v58 + 193) |= 0x40u;
                      *((_QWORD *)v29 + 4) = v49;
                      nullsub_4();
                    }
                    if ( !v62 )
                    {
                      if ( (*(_BYTE *)(v58 + 192) & 2) != 0 )
                      {
                        while ( *(_BYTE *)(v31 + 140) == 12 )
                          v31 = *(_QWORD *)(v31 + 160);
                        sub_5F7FF0(v31);
                        if ( *(_QWORD *)(*(_QWORD *)(v31 + 168) + 184LL) == v32 )
                          goto LABEL_32;
                      }
LABEL_65:
                      if ( v32 )
                      {
                        sub_6E1D20(v32);
                        *(_BYTE *)(v32 + 193) |= 0x40u;
                      }
                      goto LABEL_33;
                    }
                  }
                  else if ( !v62 )
                  {
                    goto LABEL_65;
                  }
LABEL_58:
                  if ( (unsigned int)sub_691630(v31, 0) )
                  {
                    v44 = sub_7D3810(unk_4D04844 == 0 ? 2 : 4);
                    if ( v44 )
                    {
                      v45 = sub_87C270(v44, v31, &v73[0]._IO_read_ptr);
                      if ( v45 )
                      {
                        if ( *(_QWORD *)(v45 + 88) == v32
                          && *(char *)(v32 + 192) >= 0
                          && (!unk_4D04478 || !(unsigned int)sub_87ADD0(v32, &v67, (char *)&v67 + 4, &v68) || !v67) )
                        {
                          v32 = 0;
                        }
                      }
                    }
                    if ( v58 && (*(_BYTE *)(v58 + 192) & 2) != 0 )
                      sub_6E1D20(v58);
                  }
                  goto LABEL_65;
                }
                if ( (unsigned int)sub_6E5430(0, v31, v38, v39, v40) )
                {
                  v32 = 0;
                  sub_6851C0(0x340u, v73);
                  goto LABEL_55;
                }
              }
              v32 = 0;
              goto LABEL_55;
            }
          }
          v59 = v36;
          v46 = sub_8D3A70(v31);
          v36 = v59;
          if ( v46 )
          {
            v47 = sub_7D3790(v59, v31);
            v36 = v59;
            v32 = v47;
            if ( v47 )
              goto LABEL_46;
          }
          goto LABEL_45;
        }
        if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 || !dword_4D04964 )
        {
          sub_684B30(0x354u, (_DWORD *)&v73[0]._lock + 1);
          v55 = v63;
          goto LABEL_28;
        }
      }
      sub_6E68E0(852, &v73[0]._IO_save_base);
      goto LABEL_18;
    }
  }
  sub_6E6870(&v73[0]._IO_save_base);
LABEL_18:
  sub_6E6260(v6);
LABEL_19:
  *((_DWORD *)v6 + 17) = v72;
  *((_WORD *)v6 + 36) = WORD2(v72);
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v6 + 68);
  v20 = *(__off64_t *)((char *)&v73[0]._offset + 4);
  *(__int64 *)((char *)v6 + 76) = *(__off64_t *)((char *)&v73[0]._offset + 4);
  *(_QWORD *)&dword_4F061D8 = v20;
  sub_6E3280(v6, &v72);
  sub_6E26D0(2, v6);
  return sub_724E30(&v69);
}
