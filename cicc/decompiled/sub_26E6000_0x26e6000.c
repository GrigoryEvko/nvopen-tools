// Function: sub_26E6000
// Address: 0x26e6000
//
void __fastcall sub_26E6000(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // rdx
  __int128 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r15
  int *v9; // rbx
  unsigned __int64 v10; // rbx
  __int64 *v11; // rax
  _QWORD *v12; // r8
  __int64 v13; // rax
  __int64 v14; // rbx
  char v15; // bl
  unsigned __int64 *v16; // r8
  unsigned __int64 v17; // rdi
  _QWORD *v18; // r9
  _QWORD *v19; // rax
  _QWORD *v20; // rsi
  _QWORD *v21; // r9
  unsigned __int64 v22; // r15
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int128 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // edx
  __int64 *v32; // rdi
  __int64 v33; // r9
  char v34; // bl
  unsigned __int64 *v35; // r8
  char v36; // al
  int *v37; // rsi
  size_t v38; // rdx
  int v39; // edi
  int v40; // r10d
  _QWORD *v41; // [rsp+0h] [rbp-130h]
  __int64 v42; // [rsp+8h] [rbp-128h]
  __int64 v43; // [rsp+10h] [rbp-120h]
  unsigned __int64 *v44; // [rsp+10h] [rbp-120h]
  int *v45; // [rsp+10h] [rbp-120h]
  _QWORD *v46; // [rsp+10h] [rbp-120h]
  __int64 v47; // [rsp+10h] [rbp-120h]
  char v48; // [rsp+10h] [rbp-120h]
  unsigned __int64 *v49; // [rsp+10h] [rbp-120h]
  __int64 v50; // [rsp+18h] [rbp-118h]
  __int64 v51; // [rsp+18h] [rbp-118h]
  __int64 v52; // [rsp+28h] [rbp-108h] BYREF
  unsigned __int64 v53; // [rsp+30h] [rbp-100h] BYREF
  int v54; // [rsp+38h] [rbp-F8h] BYREF
  unsigned __int64 v55; // [rsp+40h] [rbp-F0h]
  int *v56; // [rsp+48h] [rbp-E8h]
  int *v57; // [rsp+50h] [rbp-E0h]
  __int64 v58; // [rsp+58h] [rbp-D8h]
  unsigned __int64 v59; // [rsp+60h] [rbp-D0h] BYREF
  int v60; // [rsp+68h] [rbp-C8h] BYREF
  unsigned __int64 v61; // [rsp+70h] [rbp-C0h]
  int *v62; // [rsp+78h] [rbp-B8h]
  int *v63; // [rsp+80h] [rbp-B0h]
  __int64 v64; // [rsp+88h] [rbp-A8h]

  v59 = sub_B2D7E0(a2, "sample-profile-suffix-elision-policy", 0x24u);
  v43 = sub_A72240((__int64 *)&v59);
  v50 = v4;
  *(_QWORD *)&v5 = sub_BD5D20(a2);
  v6 = sub_C16140(v5, v43, v50);
  v8 = v7;
  if ( v6 )
  {
    v9 = (int *)v6;
    sub_C7D030(&v59);
    sub_C7D280((int *)&v59, v9, v8);
    sub_C7D290(&v59, &v53);
    v8 = v53;
  }
  v10 = *(_QWORD *)(a1 + 48);
  v59 = v8;
  v11 = sub_C1DD00((_QWORD *)(a1 + 40), v8 % v10, &v59, v8);
  v12 = (_QWORD *)(a1 + 40);
  if ( v11 )
  {
    v13 = *v11;
    if ( v13 )
    {
      v14 = v13 + 16;
      goto LABEL_6;
    }
  }
  if ( LOBYTE(qword_4FF8120[17]) )
  {
    v17 = *(_QWORD *)(a1 + 208);
    v18 = *(_QWORD **)(*(_QWORD *)(a1 + 200) + 8 * (a2 % v17));
    if ( v18 )
    {
      v19 = (_QWORD *)*v18;
      if ( a2 == *(_QWORD *)(*v18 + 8LL) )
      {
LABEL_24:
        v21 = (_QWORD *)*v18;
        if ( v21 )
        {
          v22 = v21[3];
          v45 = (int *)v21[2];
          if ( v45 )
          {
            v41 = v21;
            sub_C7D030(&v59);
            sub_C7D280((int *)&v59, v45, v22);
            sub_C7D290(&v59, &v53);
            v22 = v53;
            v10 = *(_QWORD *)(a1 + 48);
            v21 = v41;
            v12 = (_QWORD *)(a1 + 40);
          }
          v46 = v21;
          v59 = v22;
          v23 = sub_C1DD00(v12, v22 % v10, &v59, v22);
          if ( v23 && *v23 )
          {
            v14 = *v23 + 16LL;
            goto LABEL_6;
          }
          if ( (_BYTE)qword_4FF8648 )
          {
            v37 = (int *)v46[2];
            v38 = 0;
            if ( v37 )
              v38 = v46[3];
            v14 = sub_26C7880(*(_QWORD **)(a1 + 8), v37, v38);
            if ( v14 )
            {
LABEL_6:
              v56 = &v54;
              v57 = &v54;
              v54 = 0;
              v55 = 0;
              v58 = 0;
              sub_26E1C80(a1, a2, (__int64)&v53);
              v60 = 0;
              v62 = &v60;
              v63 = &v60;
              v61 = 0;
              v64 = 0;
              sub_26E39C0(a1, v14, &v59);
              if ( LOBYTE(qword_4FF8040[17]) || LOBYTE(qword_4FF7F60[17]) )
                sub_26E5BF0(a1, a2, (__int64)&v53, &v59, 0);
              if ( !LOBYTE(qword_4FF8200[17]) )
                goto LABEL_16;
              if ( !unk_4F838D4 )
              {
                v15 = qword_4FF8120[17];
LABEL_12:
                v44 = (unsigned __int64 *)sub_26E0930(a1, a2);
                sub_26E2F60(a1, a2, (__int64)&v53, (__int64)&v59, v44, 1, v15);
                v16 = v44;
                goto LABEL_13;
              }
              v42 = *(_QWORD *)(a1 + 24);
              v52 = sub_B2D7E0(a2, "sample-profile-suffix-elision-policy", 0x24u);
              v47 = sub_A72240(&v52);
              v51 = v24;
              *(_QWORD *)&v25 = sub_BD5D20(a2);
              v26 = sub_C16140(v25, v47, v51);
              v28 = sub_B2F650(v26, v27);
              v29 = *(unsigned int *)(v42 + 24);
              v30 = *(_QWORD *)(v42 + 8);
              if ( (_DWORD)v29 )
              {
                v31 = (v29 - 1) & (((0xBF58476D1CE4E5B9LL * v28) >> 31) ^ (484763065 * v28));
                v32 = (__int64 *)(v30 + 24LL * v31);
                v33 = *v32;
                if ( v28 == *v32 )
                {
LABEL_32:
                  if ( v32 != (__int64 *)(v30 + 24 * v29) && (*(_BYTE *)(a2 + 32) & 0xF) != 1 )
                  {
                    if ( v32[2] != *(_QWORD *)(v14 + 8) )
                      goto LABEL_35;
                    goto LABEL_38;
                  }
                }
                else
                {
                  v39 = 1;
                  while ( v33 != -1 )
                  {
                    v40 = v39 + 1;
                    v31 = (v29 - 1) & (v39 + v31);
                    v32 = (__int64 *)(v30 + 24LL * v31);
                    v33 = *v32;
                    if ( v28 == *v32 )
                      goto LABEL_32;
                    v39 = v40;
                  }
                }
              }
              if ( (unsigned __int8)sub_B2D620(a2, "profile-checksum-mismatch", 0x19u) )
              {
LABEL_35:
                v15 = qword_4FF8120[17];
                if ( *(_DWORD *)(a1 + 32) == 1 )
                  sub_B2CD60(a2, "profile-checksum-mismatch", 0x19u, 0, 0);
                goto LABEL_12;
              }
LABEL_38:
              v34 = unk_4F838D4 ^ 1;
              v48 = qword_4FF8120[17];
              v35 = (unsigned __int64 *)sub_26E0930(a1, a2);
              v36 = v48;
              v49 = v35;
              sub_26E2F60(a1, a2, (__int64)&v53, (__int64)&v59, v35, v34, v36);
              v16 = v49;
              if ( !v34 )
              {
LABEL_16:
                sub_26E0760(v61);
                sub_26E0760(v55);
                return;
              }
LABEL_13:
              if ( LOBYTE(qword_4FF8040[17]) || LOBYTE(qword_4FF7F60[17]) )
                sub_26E5BF0(a1, a2, (__int64)&v53, &v59, v16);
              goto LABEL_16;
            }
          }
        }
      }
      else
      {
        while ( 1 )
        {
          v20 = (_QWORD *)*v19;
          if ( !*v19 )
            break;
          v18 = v19;
          if ( a2 % v17 != v20[1] % v17 )
            break;
          v19 = (_QWORD *)*v19;
          if ( a2 == v20[1] )
            goto LABEL_24;
        }
      }
    }
  }
}
