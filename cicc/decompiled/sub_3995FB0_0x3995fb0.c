// Function: sub_3995FB0
// Address: 0x3995fb0
//
void __fastcall sub_3995FB0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  __int64 *v5; // rax
  __int64 v6; // r12
  char *v7; // rax
  int v8; // r8d
  int v9; // r9d
  char v10; // al
  __int64 v11; // rax
  unsigned int v12; // ecx
  __int64 *v13; // rsi
  __int64 v14; // rdi
  unsigned int v15; // edi
  __int64 *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // ecx
  int v20; // esi
  __int64 v21; // rdi
  unsigned int v22; // ecx
  int v23; // r8d
  __int64 v24; // rdx
  __int64 v25; // rax
  const char *v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rdx
  unsigned int v29; // r13d
  __int64 v30; // rdi
  unsigned int v31; // r12d
  __int64 v32; // rbx
  __int64 v33; // r14
  __int64 v34; // rdx
  char *v35; // r12
  bool v36; // cc
  char v37; // al
  int v38; // edx
  int v39; // edx
  __int64 v40; // rsi
  unsigned int v41; // eax
  __int64 *v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rdi
  void (*v53)(); // r11
  __int64 v54; // rdi
  unsigned int v55; // r9d
  int v56; // r8d
  int v57; // edi
  int v58; // r8d
  int v59; // esi
  int v60; // r10d
  int v61; // r11d
  __int64 *v62; // r10
  int v63; // edx
  unsigned int v64; // eax
  __int64 v65; // rsi
  __int64 *v66; // r8
  int v67; // r8d
  unsigned int v68; // r12d
  __int64 *v69; // rdi
  __int64 v70; // rcx
  unsigned int v71; // [rsp+10h] [rbp-D0h]
  __int64 v73; // [rsp+18h] [rbp-C8h]
  unsigned int v74; // [rsp+20h] [rbp-C0h]
  char v75; // [rsp+2Bh] [rbp-B5h]
  unsigned int v76; // [rsp+2Ch] [rbp-B4h]
  const char *v77; // [rsp+30h] [rbp-B0h]
  __int64 v78; // [rsp+38h] [rbp-A8h]
  __int64 v79; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 v80; // [rsp+48h] [rbp-98h]
  __int64 v81; // [rsp+50h] [rbp-90h]
  __int64 v82; // [rsp+58h] [rbp-88h]
  _BYTE *v83; // [rsp+60h] [rbp-80h] BYREF
  __int64 v84; // [rsp+68h] [rbp-78h]
  _BYTE v85[112]; // [rsp+70h] [rbp-70h] BYREF

  v83 = v85;
  v84 = 0x800000000LL;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v4 = sub_15C70A0(a2 + 64);
  v5 = (__int64 *)sub_1E15F70(a2);
  v6 = sub_1626D20(*v5);
  v7 = (char *)sub_16D40F0((__int64)qword_4FBB450);
  if ( v7 )
    v10 = *v7;
  else
    v10 = qword_4FBB450[2];
  v75 = v10 & (v6 != 0);
  if ( v75 )
    v75 = *(_DWORD *)(*(_QWORD *)(v6 + 8 * (5LL - *(unsigned int *)(v6 + 8))) + 36LL) == 3;
  v11 = (unsigned int)v84;
  if ( v4 )
  {
    while ( 1 )
    {
LABEL_6:
      if ( (_DWORD)v82 )
      {
        v9 = v82 - 1;
        v8 = v80;
        v12 = (v82 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v13 = (__int64 *)(v80 + 8LL * v12);
        v14 = *v13;
        if ( *v13 == v4 )
        {
LABEL_8:
          if ( v13 != (__int64 *)(v80 + 8LL * (unsigned int)v82) )
            goto LABEL_21;
        }
        else
        {
          v59 = 1;
          while ( v14 != -8 )
          {
            v60 = v59 + 1;
            v12 = v9 & (v59 + v12);
            v13 = (__int64 *)(v80 + 8LL * v12);
            v14 = *v13;
            if ( *v13 == v4 )
              goto LABEL_8;
            v59 = v60;
          }
        }
      }
      if ( !*(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8)) )
        goto LABEL_21;
      if ( HIDWORD(v84) <= (unsigned int)v11 )
      {
        sub_16CD150((__int64)&v83, v85, 0, 8, v8, v9);
        v11 = (unsigned int)v84;
      }
      *(_QWORD *)&v83[8 * v11] = v4;
      v11 = (unsigned int)(v84 + 1);
      LODWORD(v84) = v84 + 1;
      if ( !(_DWORD)v82 )
        break;
      v9 = v82 - 1;
      v15 = (v82 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v16 = (__int64 *)(v80 + 8LL * v15);
      v17 = *v16;
      if ( *v16 != v4 )
      {
        v61 = 1;
        v62 = 0;
        while ( v17 != -8 )
        {
          if ( v17 != -16 || v62 )
            v16 = v62;
          v15 = v9 & (v61 + v15);
          v17 = *(_QWORD *)(v80 + 8LL * v15);
          if ( v17 == v4 )
            goto LABEL_14;
          ++v61;
          v62 = v16;
          v16 = (__int64 *)(v80 + 8LL * v15);
        }
        if ( !v62 )
          v62 = v16;
        ++v79;
        v63 = v81 + 1;
        if ( 4 * ((int)v81 + 1) < (unsigned int)(3 * v82) )
        {
          if ( (int)v82 - HIDWORD(v81) - v63 <= (unsigned int)v82 >> 3 )
          {
            sub_3995E00((__int64)&v79, v82);
            if ( !(_DWORD)v82 )
            {
LABEL_115:
              LODWORD(v81) = v81 + 1;
              BUG();
            }
            v67 = 1;
            v68 = (v82 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v62 = (__int64 *)(v80 + 8LL * v68);
            v63 = v81 + 1;
            v69 = 0;
            v70 = *v62;
            if ( *v62 != v4 )
            {
              while ( v70 != -8 )
              {
                if ( v70 == -16 && !v69 )
                  v69 = v62;
                v9 = v67 + 1;
                v68 = (v82 - 1) & (v67 + v68);
                v62 = (__int64 *)(v80 + 8LL * v68);
                v70 = *v62;
                if ( *v62 == v4 )
                  goto LABEL_80;
                ++v67;
              }
              if ( v69 )
                v62 = v69;
            }
          }
          goto LABEL_80;
        }
LABEL_84:
        sub_3995E00((__int64)&v79, 2 * v82);
        if ( !(_DWORD)v82 )
          goto LABEL_115;
        v64 = (v82 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v62 = (__int64 *)(v80 + 8LL * v64);
        v65 = *v62;
        v63 = v81 + 1;
        if ( *v62 != v4 )
        {
          v9 = 1;
          v66 = 0;
          while ( v65 != -8 )
          {
            if ( !v66 && v65 == -16 )
              v66 = v62;
            v64 = (v82 - 1) & (v9 + v64);
            v62 = (__int64 *)(v80 + 8LL * v64);
            v65 = *v62;
            if ( *v62 == v4 )
              goto LABEL_80;
            ++v9;
          }
          if ( v66 )
            v62 = v66;
        }
LABEL_80:
        LODWORD(v81) = v63;
        if ( *v62 != -8 )
          --HIDWORD(v81);
        *v62 = v4;
        v11 = (unsigned int)v84;
      }
LABEL_14:
      if ( !v75 )
        goto LABEL_21;
      if ( *(_DWORD *)(v4 + 8) != 2 )
        goto LABEL_21;
      v18 = *(_QWORD *)(v4 - 8);
      if ( !v18 )
        goto LABEL_21;
      v19 = *(_DWORD *)(a1 + 6576);
      if ( !v19 )
        goto LABEL_21;
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 6560);
      v22 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v8 = v22;
      v4 = *(_QWORD *)(v21 + 8LL * v22);
      if ( v18 != v4 )
      {
        v23 = 1;
        while ( v4 != -8 )
        {
          v9 = v23 + 1;
          v22 = v20 & (v23 + v22);
          v8 = v22;
          v4 = *(_QWORD *)(v21 + 8LL * v22);
          if ( v18 == v4 )
            goto LABEL_6;
          v23 = v9;
        }
        goto LABEL_21;
      }
    }
    ++v79;
    goto LABEL_84;
  }
LABEL_21:
  v24 = (unsigned int)v11;
  v73 = 8LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL) + 8LL) + 1164LL);
  if ( (_DWORD)v11 )
  {
    while ( 1 )
    {
      v32 = 0;
      v33 = *(_QWORD *)&v83[8 * v24 - 8];
      LODWORD(v84) = v11 - 1;
      v34 = *(unsigned int *)(v33 + 8);
      v35 = *(char **)(v33 - 8 * v34);
      if ( (_DWORD)v34 == 2 )
        v32 = *(_QWORD *)(v33 - 8);
      v27 = *(_QWORD *)(v33 - 8 * v34);
      if ( *v35 == 15 )
        break;
      v25 = *(_QWORD *)&v35[-8 * *((unsigned int *)v35 + 2)];
      if ( v25 )
      {
        v26 = *(const char **)(*(_QWORD *)&v35[-8 * *((unsigned int *)v35 + 2)] - 8LL * *(unsigned int *)(v25 + 8));
        if ( v26 )
          goto LABEL_25;
LABEL_40:
        v28 = 0;
        goto LABEL_26;
      }
      v28 = 0;
      v26 = byte_3F871B3;
LABEL_26:
      v29 = *(_DWORD *)(v33 + 4);
      v77 = v26;
      v78 = v28;
      v76 = *(unsigned __int16 *)(v33 + 2);
      if ( v29 )
      {
        v36 = (unsigned __int16)sub_398C0A0(a1) <= 3u;
        v37 = *v35;
        if ( v36 || v37 != 19 )
        {
          v74 = 0;
          v30 = *(_QWORD *)(*(_QWORD *)(a1 + 4208) + v73);
          if ( v37 == 15 )
            goto LABEL_30;
        }
        else
        {
          v74 = *((_DWORD *)v35 + 6);
          v30 = *(_QWORD *)(*(_QWORD *)(a1 + 4208) + v73);
        }
      }
      else
      {
        v30 = *(_QWORD *)(*(_QWORD *)(a1 + 4208) + v73);
        if ( *v35 == 15 )
        {
          v74 = 0;
          v31 = sub_39CC330(v30, v35);
          goto LABEL_33;
        }
        v74 = 0;
      }
      v27 = *(_QWORD *)&v35[-8 * *((unsigned int *)v35 + 2)];
LABEL_30:
      v31 = sub_39CC330(v30, v27);
      if ( LOBYTE(qword_5056180[20]) && v29 )
        (*(void (__fastcall **)(_QWORD, const char *, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 280LL))(
          *(_QWORD *)(a1 + 8),
          v77,
          v78,
          v29,
          0);
LABEL_33:
      if ( !v75 )
        goto LABEL_34;
      v38 = *(_DWORD *)(a1 + 6576);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a1 + 6560);
        v41 = v39 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v42 = (__int64 *)(v40 + 8LL * v41);
        v43 = *v42;
        if ( v33 == *v42 )
        {
LABEL_46:
          *v42 = -16;
          --*(_DWORD *)(a1 + 6568);
          ++*(_DWORD *)(a1 + 6572);
        }
        else
        {
          v54 = *v42;
          v55 = v39 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
          v56 = 1;
          while ( v54 != -8 )
          {
            v55 = v39 & (v56 + v55);
            v54 = *(_QWORD *)(v40 + 8LL * v55);
            if ( v33 == v54 )
            {
              v57 = 1;
              while ( v43 != -8 )
              {
                v58 = v57 + 1;
                v41 = v39 & (v57 + v41);
                v42 = (__int64 *)(v40 + 8LL * v41);
                v43 = *v42;
                if ( v33 == *v42 )
                  goto LABEL_46;
                v57 = v58;
              }
              break;
            }
            ++v56;
          }
        }
      }
      if ( v32 )
      {
        v44 = *(_QWORD *)(v32 - 8LL * *(unsigned int *)(v32 + 8));
        if ( *(_BYTE *)v44 != 15 )
          v44 = *(_QWORD *)(v44 - 8LL * *(unsigned int *)(v44 + 8));
        v71 = sub_39CC330(*(_QWORD *)(*(_QWORD *)(a1 + 4208) + v73), v44);
        v45 = sub_15AB1E0(*(_BYTE **)(v33 - 8LL * *(unsigned int *)(v33 + 8)));
        v46 = *(unsigned int *)(v45 + 8);
        v47 = *(_QWORD *)(v45 + 8 * (3 - v46));
        if ( v47 )
        {
          v48 = sub_161E970(*(_QWORD *)(v45 + 8 * (3 - v46)));
          v50 = v49;
          v47 = v48;
        }
        else
        {
          v50 = 0;
        }
        v51 = sub_39A1860(a1 + 4232, *(_QWORD *)(a1 + 8), v47, v50);
        v52 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
        v53 = *(void (**)())(*(_QWORD *)v52 + 592LL);
        if ( v53 == nullsub_590 )
          goto LABEL_35;
        ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, const char *, __int64))v53)(
          v52,
          v31,
          v29,
          v76,
          v71,
          *(unsigned int *)(v32 + 4),
          *(unsigned __int16 *)(v32 + 2),
          *(_QWORD *)(v51 + 8),
          a3,
          0,
          v74,
          v77,
          v78);
        v24 = (unsigned int)v84;
        LODWORD(v11) = v84;
        if ( !(_DWORD)v84 )
          goto LABEL_54;
      }
      else
      {
LABEL_34:
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, const char *, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 584LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
          v31,
          v29,
          v76,
          a3,
          0,
          v74,
          v77,
          v78);
LABEL_35:
        v24 = (unsigned int)v84;
        LODWORD(v11) = v84;
        if ( !(_DWORD)v84 )
          goto LABEL_54;
      }
    }
    v26 = *(const char **)&v35[-8 * *((unsigned int *)v35 + 2)];
    if ( v26 )
    {
LABEL_25:
      v27 = *(_QWORD *)(v33 - 8 * v34);
      v26 = (const char *)sub_161E970((__int64)v26);
      goto LABEL_26;
    }
    goto LABEL_40;
  }
LABEL_54:
  j___libc_free_0(v80);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
}
