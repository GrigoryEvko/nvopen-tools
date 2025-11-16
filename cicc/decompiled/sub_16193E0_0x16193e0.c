// Function: sub_16193E0
// Address: 0x16193e0
//
__int64 __fastcall sub_16193E0(__int64 a1, char *a2)
{
  unsigned __int64 v2; // r15
  __int64 *v3; // rbx
  __int64 *i; // r13
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r12
  int v11; // r14d
  __int64 v12; // rdi
  int v13; // r13d
  unsigned int v14; // r12d
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r12
  _QWORD *v19; // rax
  unsigned __int64 v20; // rsi
  _QWORD *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  _QWORD *v24; // rax
  unsigned __int8 v25; // r14
  __int64 v26; // rdi
  unsigned int (__fastcall *v27)(_QWORD, _QWORD); // rax
  unsigned int v28; // r12d
  int v29; // r12d
  int v30; // r14d
  __int64 v31; // r13
  __int64 v32; // rdi
  __int64 v33; // r12
  __int64 v34; // rbx
  int v35; // r14d
  __int64 v36; // r13
  unsigned int (__fastcall *v37)(_QWORD, _QWORD); // rax
  __int64 *v38; // rbx
  __int64 *j; // r12
  __int64 v40; // rdi
  __int64 (*v41)(); // rax
  __int64 *v43; // r10
  char v44; // r12
  __int64 *v45; // r14
  __int64 v46; // rbx
  _QWORD *v47; // rax
  char v48; // di
  _BOOL4 v49; // r11d
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // [rsp+8h] [rbp-D8h]
  __int64 v60; // [rsp+10h] [rbp-D0h]
  unsigned int v62; // [rsp+20h] [rbp-C0h]
  unsigned int v63; // [rsp+24h] [rbp-BCh]
  _QWORD *v64; // [rsp+30h] [rbp-B0h]
  __int64 v65; // [rsp+38h] [rbp-A8h]
  __int64 v66; // [rsp+38h] [rbp-A8h]
  _BOOL4 v67; // [rsp+40h] [rbp-A0h]
  _QWORD *v68; // [rsp+40h] [rbp-A0h]
  unsigned __int8 v69; // [rsp+48h] [rbp-98h]
  __int64 *v70; // [rsp+48h] [rbp-98h]
  _QWORD *v71; // [rsp+60h] [rbp-80h]
  unsigned __int8 v72; // [rsp+6Ah] [rbp-76h]
  char v73; // [rsp+6Bh] [rbp-75h]
  unsigned int v74; // [rsp+6Ch] [rbp-74h]
  __int64 v75; // [rsp+78h] [rbp-68h] BYREF
  _QWORD v76[12]; // [rsp+80h] [rbp-60h] BYREF

  v2 = (unsigned __int64)a2;
  if ( unk_4F9E388 )
  {
    v52 = *((_QWORD *)a2 + 22);
    v53 = *((_QWORD *)a2 + 23);
    a2 = "<unnamed>";
    v76[0] = v52;
    v76[1] = v53;
    if ( sub_16D20C0(v76, "<unnamed>", 9, 0) == -1 )
    {
      v56 = sub_16E8C20(v76, "<unnamed>", v54, v55);
      v57 = sub_1263B40(v56, "Timing info for ");
      v58 = sub_1549FF0(v57, *(const char **)(v2 + 176), *(_QWORD *)(v2 + 184));
      a2 = "\n";
      sub_1263B40(v58, "\n");
    }
    if ( unk_4F9E388 && !qword_4F9E390 )
    {
      if ( !qword_4F9E2A0 )
      {
        a2 = (char *)sub_160CF40;
        sub_16C1EA0(&qword_4F9E2A0, sub_160CF40, sub_160D890);
      }
      qword_4F9E390 = qword_4F9E2A0;
    }
  }
  sub_1615700(a1 + 568, a2);
  sub_160E900(a1 + 568);
  v3 = *(__int64 **)(a1 + 824);
  v72 = 0;
  for ( i = &v3[*(unsigned int *)(a1 + 832)];
        i != v3;
        v72 |= ((__int64 (__fastcall *)(__int64, unsigned __int64))v6)(v5, v2) )
  {
    while ( 1 )
    {
      v5 = *v3;
      v6 = *(__int64 (**)())(*(_QWORD *)*v3 + 24LL);
      if ( v6 != sub_134C070 )
        break;
      if ( i == ++v3 )
        goto LABEL_7;
    }
    ++v3;
  }
LABEL_7:
  sub_1616C40(a1 + 568);
  if ( *(_DWORD *)(a1 + 608) )
  {
    v62 = 0;
    while ( 1 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v62);
      if ( !v7 )
      {
        MEMORY[0xA8] = *(_QWORD *)(a1 + 168);
        BUG();
      }
      *(_QWORD *)(v7 + 8) = *(_QWORD *)(a1 + 168);
      v8 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v62);
      if ( !v8 )
        BUG();
      v9 = *(_QWORD *)(v8 + 440);
      v10 = *(_QWORD *)(v8 + 448);
      v69 = 0;
      v11 = 0;
      if ( v9 != v10 )
      {
        do
        {
          v12 = *(_QWORD *)(v9 + 8);
          v9 += 16;
          v11 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v12 + 24LL))(v12, v2);
        }
        while ( v10 != v9 );
        v69 = v11;
      }
      v13 = v69;
      v14 = 0;
      if ( *(_DWORD *)(v8 + 32) )
      {
        do
        {
          v15 = v14++;
          v16 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v15);
          v13 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v16 + 24LL))(v16, v2);
        }
        while ( *(_DWORD *)(v8 + 32) > v14 );
        v69 = v13;
      }
      v17 = sub_16033E0(*(_QWORD *)v2);
      v73 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v17 + 24LL))(v17, "size-info", 9);
      if ( *(_DWORD *)(v8 + 32) )
        break;
LABEL_49:
      v33 = *(_QWORD *)(v8 + 440);
      v34 = *(_QWORD *)(v8 + 448);
      if ( v33 != v34 )
      {
        v35 = v69;
        do
        {
          v36 = *(_QWORD *)(v33 + 8);
          v33 += 16;
          sub_160FA70(v36);
          v35 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v36 + 32LL))(v36, v2);
        }
        while ( v34 != v33 );
        v69 = v35;
      }
      v72 |= v69;
      sub_16027A0(*(_QWORD *)v2);
      v37 = *(unsigned int (__fastcall **)(_QWORD, _QWORD))(a1 + 1304);
      if ( (!v37 || !v37(*(_QWORD *)(a1 + 1312), 0)) && ++v62 < *(_DWORD *)(a1 + 608) )
        continue;
      goto LABEL_56;
    }
    v74 = 0;
    v63 = 0;
LABEL_19:
    v18 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8LL * v74);
    sub_160F160(v8, v18, 0, 5, *(const char **)(v2 + 176), *(_QWORD *)(v2 + 184));
    sub_1615D60(v8, (__int64 *)v18);
    sub_1614C80(v8, v18);
    sub_16C6860(v76);
    v76[2] = v18;
    v76[3] = 0;
    v76[0] = &unk_49ED7C0;
    v76[4] = v2;
    v19 = sub_1612E30((_QWORD *)v18);
    v71 = v19;
    if ( v19 )
      sub_16D7910(v19);
    v20 = v18;
    sub_1403F30(&v75, (_QWORD *)v18, *(_QWORD *)(v8 + 8));
    if ( v73 )
    {
      v20 = v2;
      v63 = sub_160E760(v8, v2);
    }
    if ( *(_BYTE *)(v18 + 152) )
    {
      v24 = *(_QWORD **)(v18 + 120);
      if ( !v24 )
      {
        v25 = 0;
        goto LABEL_32;
      }
      v20 = v18 + 112;
      do
      {
        while ( 1 )
        {
          v22 = v24[2];
          v21 = (_QWORD *)v24[3];
          if ( v24[4] >= v2 )
            break;
          v24 = (_QWORD *)v24[3];
          if ( !v21 )
            goto LABEL_29;
        }
        v20 = (unsigned __int64)v24;
        v24 = (_QWORD *)v24[2];
      }
      while ( v22 );
LABEL_29:
      v25 = 0;
      if ( v18 + 112 == v20 || *(_QWORD *)(v20 + 32) > v2 )
        goto LABEL_32;
    }
    v20 = v2;
    v25 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v18 + 144LL))(v18, v2);
    if ( !v25 )
      goto LABEL_32;
    v43 = *(__int64 **)(v18 + 32);
    v70 = *(__int64 **)(v18 + 40);
    if ( v43 == v70 )
      goto LABEL_76;
    v60 = v8;
    v59 = v18;
    v44 = v25;
    v45 = v43;
    while ( 1 )
    {
      v46 = *v45;
      v21 = *(_QWORD **)(*v45 + 120);
      v22 = *v45 + 112;
      if ( !v21 )
        break;
      while ( 1 )
      {
        v20 = v21[4];
        v47 = (_QWORD *)v21[3];
        v48 = 0;
        if ( v2 < v20 )
        {
          v47 = (_QWORD *)v21[2];
          v48 = v44;
        }
        if ( !v47 )
          break;
        v21 = v47;
      }
      if ( v48 )
      {
        if ( *(_QWORD **)(v46 + 128) != v21 )
          goto LABEL_78;
LABEL_72:
        v49 = 1;
        if ( (_QWORD *)v22 == v21 )
        {
LABEL_73:
          v64 = v21;
          v65 = v22;
          v67 = v49;
          v50 = sub_22077B0(40);
          *(_QWORD *)(v50 + 32) = v2;
          v20 = v50;
          sub_220F040(v67, v50, v64, v65);
          ++*(_QWORD *)(v46 + 144);
          goto LABEL_74;
        }
LABEL_80:
        v49 = v2 < v21[4];
        goto LABEL_73;
      }
      if ( v2 > v20 )
        goto LABEL_72;
LABEL_74:
      if ( v70 == ++v45 )
      {
        v25 = v44;
        v8 = v60;
        v18 = v59;
LABEL_76:
        v69 = v25;
LABEL_32:
        if ( v73 )
        {
          v20 = v18;
          sub_160FF80(v8, v18, v2, v63);
        }
        v26 = v75;
        if ( v75 )
        {
          if ( v25 )
          {
            v20 = 2;
            (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64))(*(_QWORD *)v75 + 56LL))(v75, 2, v21, v22);
            v26 = v75;
          }
          (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD *, __int64))(*(_QWORD *)v26 + 48LL))(
            v26,
            v20,
            v21,
            v22);
        }
        if ( v71 )
          sub_16D7950(v71, v20, v21);
        v76[0] = &unk_49ED7C0;
        nullsub_616(v76, v20, v21, v22, v23);
        if ( v25 )
          sub_160F160(v8, v18, 1, 5, *(const char **)(v2 + 176), *(_QWORD *)(v2 + 184));
        sub_1615E90(v8, (__int64 *)v18);
        sub_1615FB0(v8, (__int64 *)v18);
        nullsub_568();
        sub_16145F0(v8, v18);
        sub_16176C0(v8, v18);
        sub_1615450(v8, v18, *(const char **)(v2 + 176), *(_QWORD *)(v2 + 184), 5);
        v27 = *(unsigned int (__fastcall **)(_QWORD, _QWORD))(v8 + 464);
        if ( v27 && v27(*(_QWORD *)(v8 + 472), 0) )
        {
          v28 = *(_DWORD *)(v8 + 32);
          goto LABEL_45;
        }
        ++v74;
        v28 = *(_DWORD *)(v8 + 32);
        if ( v28 <= v74 )
        {
LABEL_45:
          v29 = v28 - 1;
          if ( v29 >= 0 )
          {
            v30 = v69;
            v31 = 8LL * v29;
            do
            {
              --v29;
              v32 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + v31);
              v31 -= 8;
              v30 |= (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v32 + 32LL))(v32, v2);
            }
            while ( v29 != -1 );
            v69 = v30;
          }
          goto LABEL_49;
        }
        goto LABEL_19;
      }
    }
    v21 = (_QWORD *)(*v45 + 112);
    if ( v22 == *(_QWORD *)(v46 + 128) )
    {
      v49 = 1;
      goto LABEL_73;
    }
LABEL_78:
    v66 = *v45 + 112;
    v68 = v21;
    v51 = sub_220EF80(v21);
    v21 = v68;
    v22 = v66;
    if ( v2 <= *(_QWORD *)(v51 + 32) )
      goto LABEL_74;
    v49 = 1;
    if ( (_QWORD *)v66 == v68 )
      goto LABEL_73;
    goto LABEL_80;
  }
LABEL_56:
  v38 = *(__int64 **)(a1 + 824);
  for ( j = &v38[*(unsigned int *)(a1 + 832)];
        j != v38;
        v72 |= ((__int64 (__fastcall *)(__int64, unsigned __int64))v41)(v40, v2) )
  {
    while ( 1 )
    {
      v40 = *v38;
      v41 = *(__int64 (**)())(*(_QWORD *)*v38 + 32LL);
      if ( v41 != sub_134C080 )
        break;
      if ( j == ++v38 )
        return v72;
    }
    ++v38;
  }
  return v72;
}
