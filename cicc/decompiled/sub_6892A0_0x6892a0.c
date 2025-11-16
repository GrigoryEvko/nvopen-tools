// Function: sub_6892A0
// Address: 0x6892a0
//
__int64 __fastcall sub_6892A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r14
  __int64 *v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r12
  char v15; // al
  __int64 v16; // r14
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  int v20; // r10d
  __int64 v21; // rbx
  int v22; // eax
  int v23; // r10d
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  int v28; // r10d
  int v29; // r14d
  __int64 v30; // rbx
  __int64 v31; // rax
  int v32; // r10d
  __int64 v33; // rax
  __int64 v34; // rdi
  _QWORD *v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  char v42; // si
  unsigned int v43; // eax
  __int64 v44; // r12
  int v45; // eax
  int v46; // eax
  unsigned int v47; // eax
  __int64 v48; // rax
  __int64 v49; // rax
  int v50; // [rsp+8h] [rbp-1C8h]
  int v51; // [rsp+10h] [rbp-1C0h]
  int v52; // [rsp+10h] [rbp-1C0h]
  __int64 v53; // [rsp+10h] [rbp-1C0h]
  int v54; // [rsp+10h] [rbp-1C0h]
  int v55; // [rsp+10h] [rbp-1C0h]
  int v56; // [rsp+10h] [rbp-1C0h]
  int v57; // [rsp+10h] [rbp-1C0h]
  _QWORD *v59; // [rsp+20h] [rbp-1B0h]
  int v60; // [rsp+20h] [rbp-1B0h]
  int v61; // [rsp+20h] [rbp-1B0h]
  __int64 v62; // [rsp+28h] [rbp-1A8h]
  int v63; // [rsp+3Ch] [rbp-194h] BYREF
  __int128 v64; // [rsp+40h] [rbp-190h] BYREF
  __int128 v65; // [rsp+50h] [rbp-180h]
  __int128 v66; // [rsp+60h] [rbp-170h]
  __int64 v67; // [rsp+84h] [rbp-14Ch]
  __int64 v68; // [rsp+8Ch] [rbp-144h]

  v6 = 0;
  v62 = 0;
  v7 = (__int64 *)*a1;
  if ( *a1 )
  {
    while ( 1 )
    {
      v13 = v7[3];
      v63 = 0;
      v14 = *(_QWORD *)(v13 + 120);
      v15 = *((_BYTE *)v7 + 32);
      if ( (v15 & 1) != 0 )
      {
        v64 = 0;
        v65 = 0;
        v66 = 0;
        if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
          BYTE10(v66) |= 1u;
        BYTE8(v66) |= 0x20u;
        v33 = sub_6327C0(v7[3], v7[1], a1[1], &v64, (__int64)v7 + 36, a6);
        v12 = v33;
        if ( (BYTE9(v66) & 4) != 0 )
        {
          v34 = *(_QWORD *)(v33 + 176);
          v6 = 1;
          if ( *(_QWORD *)(v34 + 16) )
            sub_734250(v34, *(_BYTE *)(unk_4D03C50 + 17LL) & 1);
        }
LABEL_8:
        if ( !v62 )
          goto LABEL_25;
        goto LABEL_9;
      }
      if ( (v15 & 4) != 0 )
      {
        v8 = v14;
        if ( (v15 & 8) == 0 )
          v8 = sub_72D2E0(v14, 0);
        v9 = sub_725A70(3);
        v10 = (__int64 *)sub_726700(24);
        *v10 = v8;
        if ( (v7[4] & 8) == 0 )
          v10 = (__int64 *)sub_73DBF0(3, v14, v10);
        *(_QWORD *)(v9 + 56) = v10;
        v11 = sub_724D50(9);
        *(_QWORD *)(v11 + 176) = v9;
        v12 = v11;
        *(_QWORD *)(v11 + 128) = v8;
        v6 = 1;
        goto LABEL_8;
      }
      v16 = v7[2];
      if ( (v15 & 2) != 0 )
      {
        if ( v16 )
        {
          v59 = 0;
          goto LABEL_15;
        }
      }
      else
      {
        v35 = (_QWORD *)v7[1];
        v59 = v35;
        if ( v35 )
        {
          v36 = *v35;
          if ( *v35 )
          {
            v37 = *(_QWORD *)(v36 + 88);
            *(_QWORD *)(v36 + 88) = v35;
            v53 = v37;
            sub_8767A0((-(__int64)((v7[4] & 8) == 0) & 0xFFFFFFFFFFFFFFE8LL) + 36, v36, (char *)v7 + 36, 1);
            *(_QWORD *)(v36 + 88) = v53;
          }
          if ( !v16 )
          {
            sub_6F8E70(v59, (char *)v7 + 36, (char *)v7 + 44, &v64, 0);
            v51 = 0;
            if ( (*((_BYTE *)v59 + 172) & 1) != 0 && (v7[4] & 8) == 0 )
            {
              v51 = 1;
              v16 = sub_8D46C0(v59[15]);
            }
            v21 = v14;
            v20 = sub_8D3410(v14);
            if ( !v20 )
              goto LABEL_20;
LABEL_54:
            v38 = sub_8D40F0(v14);
            v20 = 1;
            v21 = v38;
LABEL_19:
            if ( v59 )
            {
LABEL_20:
              if ( *((_BYTE *)v59 + 177) != 5 )
                goto LABEL_21;
              v30 = 0;
              if ( v63 )
                goto LABEL_77;
LABEL_58:
              if ( (*((_BYTE *)v59 + 172) & 1) != 0 && (v7[4] & 8) != 0 && BYTE1(v65) == 1 )
              {
                v56 = v20;
                v45 = sub_6ED0A0(&v64);
                v20 = v56;
                if ( !v45 )
                {
                  v46 = sub_8D2EF0(v64);
                  v20 = v56;
                  if ( !v46 )
                  {
                    sub_6FF5E0(&v64, (char *)v7 + 36);
                    v20 = v56;
                  }
                }
              }
LABEL_59:
              v54 = v20;
              sub_843C40((unsigned int)&v64, v14, 0, 0, 1, 0, 1740);
              v24 = sub_6EAFA0(3);
              v40 = sub_6F6F40(&v64, 0);
              v32 = v54;
              *(_QWORD *)(v24 + 56) = v40;
              goto LABEL_60;
            }
LABEL_21:
            v50 = v20;
            v22 = sub_8D3A70(v21);
            v23 = v50;
            if ( v22 )
              goto LABEL_32;
            if ( v50 )
            {
              v63 = 1;
              v24 = sub_6EAFA0(7);
              goto LABEL_24;
            }
            v20 = v63;
            v30 = 0;
            if ( v63 )
            {
LABEL_77:
              v24 = sub_6EAFA0(7);
              goto LABEL_24;
            }
LABEL_71:
            if ( v59 )
              goto LABEL_58;
            goto LABEL_59;
          }
LABEL_15:
          v17 = sub_73E870();
          if ( (unsigned int)sub_8D32E0(*(_QWORD *)(v16 + 120)) )
          {
            v18 = sub_73E8A0(v17, v16);
            v19 = sub_73DDB0(v18);
          }
          else
          {
            v19 = sub_73DE50(v17, v16);
          }
          sub_6E7150(v19, &v64);
          v67 = *(__int64 *)((char *)v7 + 36);
          v68 = *(__int64 *)((char *)v7 + 44);
          v20 = sub_8D3410(v14);
          if ( !v20 )
          {
            v51 = 0;
            v21 = v14;
            v16 = 0;
            goto LABEL_19;
          }
LABEL_69:
          v51 = 0;
          v16 = 0;
          goto LABEL_54;
        }
        if ( v16 )
          goto LABEL_15;
      }
      sub_6E6260(&v64);
      v51 = sub_8D3410(v14);
      if ( v51 )
      {
        v59 = 0;
        goto LABEL_69;
      }
      if ( !(unsigned int)sub_8D3A70(v14) )
      {
        v20 = v63;
        v30 = 0;
        if ( v63 )
          goto LABEL_77;
        goto LABEL_59;
      }
      if ( *(_BYTE *)(v14 + 140) == 12 )
      {
        v23 = 0;
        v21 = v14;
        v16 = 0;
        v59 = 0;
        do
        {
          v21 = *(_QWORD *)(v21 + 160);
LABEL_32:
          ;
        }
        while ( *(_BYTE *)(v21 + 140) == 12 );
        if ( v51 )
        {
          v26 = 0;
          if ( (*(_BYTE *)(v16 + 140) & 0xFB) == 8 )
          {
            v57 = v23;
            v47 = sub_8D4C10(v16, dword_4F077C4 != 2);
            v23 = v57;
            v26 = v47;
          }
          goto LABEL_36;
        }
      }
      else
      {
        v59 = 0;
        v21 = v14;
        v23 = 0;
      }
      v26 = 0;
      if ( (*(_BYTE *)(v64 + 140) & 0xFB) == 8 )
      {
        v55 = v23;
        v43 = sub_8D4C10(v64, dword_4F077C4 != 2);
        v23 = v55;
        v26 = v43;
      }
LABEL_36:
      v52 = v23;
      v27 = sub_6EB190(v21, v26, 0, (char *)v7 + 36, &v63, 1);
      v28 = v52;
      v29 = v27;
      if ( !v27 )
      {
        if ( !v63 )
        {
          v24 = sub_6EAFA0(0);
          goto LABEL_24;
        }
        if ( !dword_4D048B8 )
          goto LABEL_77;
        v48 = sub_6EB2F0(v21, v21, (char *)v7 + 36, 0);
        v20 = v52;
        v30 = v48;
        if ( v63 )
        {
LABEL_89:
          v24 = sub_6EAFA0(7);
          if ( !v30 )
            goto LABEL_24;
          v32 = 0;
          v41 = unk_4D03C50;
          if ( *(_BYTE *)(unk_4D03C50 + 16LL) <= 3u )
            goto LABEL_24;
LABEL_62:
          *(_QWORD *)(v24 + 16) = v30;
          v42 = *(_BYTE *)(v41 + 17);
          if ( (v42 & 4) != 0 )
          {
            *(_BYTE *)(v30 + 193) |= 0x40u;
            v42 = *(_BYTE *)(v41 + 17);
          }
          v61 = v32;
          sub_734250(v24, v42 & 1);
          v32 = v61;
          goto LABEL_65;
        }
        goto LABEL_71;
      }
      if ( dword_4D048B8 )
      {
        v49 = sub_6EB2F0(v21, v21, (char *)v7 + 36, 0);
        v28 = v52;
        v30 = v49;
        if ( v63 )
          goto LABEL_89;
      }
      else
      {
        v30 = 0;
        if ( v63 )
          goto LABEL_77;
      }
      v60 = v28;
      v31 = sub_6F5430(v29, 0, v14, 0, 1, 1, 0, 0, 1, 0, (__int64)v7 + 36);
      v32 = v60;
      v24 = v31;
LABEL_60:
      if ( v30 )
      {
        v41 = unk_4D03C50;
        if ( *(_BYTE *)(unk_4D03C50 + 16LL) > 3u )
          goto LABEL_62;
      }
LABEL_65:
      if ( v32 )
        v24 = sub_6EB060(v24, v14, 0);
LABEL_24:
      v25 = sub_724D50(9);
      *(_QWORD *)(v25 + 176) = v24;
      v12 = v25;
      v6 = 1;
      *(_QWORD *)(v25 + 128) = v14;
      if ( !v62 )
      {
LABEL_25:
        v62 = sub_724D50(10);
        *(_QWORD *)(v62 + 128) = a1[1];
      }
LABEL_9:
      sub_72A690(v12, v62, 0, 0);
      v7 = (__int64 *)*v7;
      if ( !v7 )
      {
        v44 = sub_6EAFA0(8);
        *(_QWORD *)(v44 + 64) = a1;
        *(_BYTE *)(v44 + 72) = *(_BYTE *)(v44 + 72) & 0xFE | v6;
        sub_72F900(v44, v62);
        goto LABEL_56;
      }
    }
  }
  v44 = sub_6EAFA0(0);
LABEL_56:
  sub_6EB360(v44, a1[1], a1[1], (char *)a1 + 44);
  sub_6EB4C0(v44);
  return v44;
}
