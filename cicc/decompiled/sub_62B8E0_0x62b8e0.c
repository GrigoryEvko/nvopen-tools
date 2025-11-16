// Function: sub_62B8E0
// Address: 0x62b8e0
//
unsigned __int16 *__fastcall sub_62B8E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rcx
  int v9; // edi
  unsigned int v10; // edx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  unsigned int v14; // edx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rax
  unsigned int v26; // edx
  int v27; // eax
  _DWORD *v28; // rax
  unsigned int v29; // r8d
  _QWORD *v30; // rcx
  _DWORD *v31; // rdi
  _DWORD *v32; // rsi
  int *v33; // r10
  int *v34; // rsi
  __int64 v35; // r11
  int v36; // r9d
  unsigned int i; // edx
  _DWORD *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rbx
  char v41; // dl
  __int64 v42; // rax
  unsigned __int16 *result; // rax
  _DWORD *v44; // rax
  _DWORD *v45; // rsi
  int *v46; // rsi
  int v47; // r9d
  unsigned int j; // edx
  _DWORD *v49; // rax
  __int64 v50; // rax
  int v51; // esi
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rax
  int v55; // edi
  __int64 v56; // rax
  __int64 v57; // rdi
  _QWORD *v58; // [rsp+10h] [rbp-F0h]
  _QWORD *v59; // [rsp+10h] [rbp-F0h]
  unsigned int v60; // [rsp+18h] [rbp-E8h]
  unsigned int v61; // [rsp+18h] [rbp-E8h]
  unsigned int v62; // [rsp+1Ch] [rbp-E4h]
  unsigned int v63; // [rsp+1Ch] [rbp-E4h]
  __int64 v64; // [rsp+20h] [rbp-E0h]
  unsigned int v65; // [rsp+28h] [rbp-D8h]
  unsigned int v66; // [rsp+28h] [rbp-D8h]
  bool v67; // [rsp+2Fh] [rbp-D1h]
  __int64 v69; // [rsp+38h] [rbp-C8h]
  int v71; // [rsp+50h] [rbp-B0h]
  unsigned int v72; // [rsp+54h] [rbp-ACh]
  __int64 v73; // [rsp+58h] [rbp-A8h]
  unsigned int v74; // [rsp+58h] [rbp-A8h]
  __int64 v75; // [rsp+68h] [rbp-98h] BYREF
  __int64 v76[4]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE v77[112]; // [rsp+90h] [rbp-70h] BYREF

  v75 = sub_72CBE0();
  v6 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v72 = dword_4F06650[0];
  v64 = *(_QWORD *)(a1 + 432);
  v67 = (*(_BYTE *)(v6 + 6) & 2) != 0;
  v7 = *(_BYTE *)(v6 + 4);
  if ( v7 != 6 )
  {
    if ( v7 == 8 )
    {
      if ( *(_BYTE *)(v6 - 772) != 6 )
      {
LABEL_5:
        v71 = 0;
        v69 = sub_72C930();
        goto LABEL_13;
      }
      v71 = unk_4D04484;
      if ( unk_4D04484 )
      {
        sub_643C90(a1);
        sub_866340();
        v71 = 1;
      }
    }
    else
    {
      if ( v7 != 9 || *(_BYTE *)(v6 - 772) != 7 )
        goto LABEL_5;
      v71 = 0;
    }
    v69 = *(_QWORD *)(v6 - 568);
    goto LABEL_13;
  }
  v69 = *(_QWORD *)(v6 + 208);
  v51 = *(_DWORD *)(qword_4CFDE38 + 8);
  v52 = *(_QWORD *)qword_4CFDE38;
  v53 = v51 & dword_4F06650[0];
  v54 = *(_QWORD *)qword_4CFDE38 + 16 * v53;
  v55 = *(_DWORD *)v54;
  if ( dword_4F06650[0] == *(_DWORD *)v54 )
  {
LABEL_80:
    v56 = *(_QWORD *)(v54 + 8);
    *(_QWORD *)(a1 + 368) = v56;
    if ( v56 )
    {
      sub_89F0F0(a1, a3, v53, v52, v72);
      *(_BYTE *)(a1 + 133) |= 8u;
      *(_QWORD *)(a1 + 368) = 0;
      *(_DWORD *)(a1 + 64) = v72;
      sub_7BDB60(1);
      v71 = 0;
      goto LABEL_13;
    }
  }
  else
  {
    while ( v55 )
    {
      v53 = v51 & (unsigned int)(v53 + 1);
      v54 = v52 + 16LL * (unsigned int)v53;
      v55 = *(_DWORD *)v54;
      if ( dword_4F06650[0] == *(_DWORD *)v54 )
        goto LABEL_80;
    }
    *(_QWORD *)(a1 + 368) = 0;
  }
  v71 = unk_4D04484;
  if ( unk_4D04484 )
  {
    sub_643C90(a1);
    sub_866340();
    v71 = 0;
  }
LABEL_13:
  while ( 1 )
  {
    sub_87A720(42, v77, &dword_4F063F8);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    sub_7B8B50(42, v77, v16, v17);
    sub_627530(a1, 0x10u, &v75, (char *)a2, v77, v69, 1u, 0, 0, 0, 0, unk_4D04488 == 0, 0, a4);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    if ( (*(_BYTE *)(a1 + 133) & 0x20) == 0 )
      break;
    if ( !*(_QWORD *)(a1 + 368) )
    {
      sub_643D30(a1);
      sub_866420(0);
      break;
    }
    v73 = *(_QWORD *)(a2 + 56);
    sub_7ADF70(v76, 0);
    sub_7AE700(unk_4F061C0 + 24LL, v72, dword_4F06650[0], 0, v76);
    sub_643D30(a1);
    sub_7BC000(v76);
    sub_89F0F0(a1, a3, v18, v19, v20);
    sub_643F80(a1, v64);
    sub_65C040(a1);
    *(_BYTE *)(a1 + 121) &= ~0x80u;
    *(_BYTE *)(a1 + 133) |= 8u;
    v21 = *(_QWORD *)&dword_4F063F8;
    *(_QWORD *)(a1 + 40) = *(_QWORD *)&dword_4F063F8;
    *(_QWORD *)(a1 + 48) = v21;
    sub_87E3B0(a2);
    *(_QWORD *)(a2 + 56) = v73;
    if ( !v71 )
      sub_7BDB60(1);
    if ( v67 )
    {
      v8 = qword_4CFDE38;
      v9 = *(_DWORD *)(qword_4CFDE38 + 8);
      v10 = v9 & v72;
      v11 = 16LL * (v9 & v72);
      v12 = *(_QWORD *)qword_4CFDE38 + v11;
      if ( *(_DWORD *)v12 )
      {
        do
        {
          v10 = v9 & (v10 + 1);
          v22 = *(_QWORD *)qword_4CFDE38 + 16LL * v10;
        }
        while ( *(_DWORD *)v22 );
        v23 = *(_QWORD *)(v12 + 8);
        *(_DWORD *)v22 = *(_DWORD *)v12;
        *(_QWORD *)(v22 + 8) = v23;
        *(_DWORD *)v12 = 0;
        v24 = *(_QWORD *)v8 + v11;
        v25 = *(_QWORD *)(a1 + 368);
        *(_DWORD *)v24 = v72;
        if ( v72 )
          *(_QWORD *)(v24 + 8) = v25;
        v26 = *(_DWORD *)(v8 + 8);
        v27 = *(_DWORD *)(v8 + 12) + 1;
        *(_DWORD *)(v8 + 12) = v27;
        if ( 2 * v27 <= v26 )
          goto LABEL_11;
        v58 = (_QWORD *)v8;
        v60 = v26;
        v65 = 2 * v26 + 2;
        v62 = 2 * v26 + 1;
        v74 = v26 + 1;
        v28 = (_DWORD *)sub_823970(16LL * v65);
        v29 = v62;
        v30 = v58;
        v31 = v28;
        if ( v65 )
        {
          v32 = &v28[4 * v62 + 4];
          do
          {
            if ( v28 )
              *v28 = 0;
            v28 += 4;
          }
          while ( v32 != v28 );
        }
        v33 = (int *)*v58;
        if ( v74 )
        {
          v34 = (int *)*v58;
          v35 = (__int64)&v33[4 * v60 + 4];
          do
          {
            while ( 1 )
            {
              v36 = *v34;
              if ( *v34 )
                break;
              v34 += 4;
              if ( (int *)v35 == v34 )
                goto LABEL_39;
            }
            for ( i = v36 & v62; ; i = v62 & (i + 1) )
            {
              v38 = &v31[4 * i];
              if ( !*v38 )
                break;
            }
            *v38 = v36;
            v39 = *((_QWORD *)v34 + 1);
            v34 += 4;
            *((_QWORD *)v38 + 1) = v39;
          }
          while ( (int *)v35 != v34 );
        }
      }
      else
      {
        v13 = *(_QWORD *)(a1 + 368);
        *(_DWORD *)v12 = v72;
        if ( v72 )
          *(_QWORD *)(v12 + 8) = v13;
        v14 = *(_DWORD *)(v8 + 8);
        v15 = *(_DWORD *)(v8 + 12) + 1;
        *(_DWORD *)(v8 + 12) = v15;
        if ( 2 * v15 <= v14 )
          goto LABEL_11;
        v59 = (_QWORD *)v8;
        v61 = v14;
        v66 = 2 * v14 + 2;
        v63 = 2 * v14 + 1;
        v74 = v14 + 1;
        v44 = (_DWORD *)sub_823970(16LL * v66);
        v29 = v63;
        v30 = v59;
        v31 = v44;
        if ( v66 )
        {
          v45 = &v44[4 * v63 + 4];
          do
          {
            if ( v44 )
              *v44 = 0;
            v44 += 4;
          }
          while ( v45 != v44 );
        }
        v33 = (int *)*v59;
        if ( v74 )
        {
          v46 = (int *)*v59;
          do
          {
            v47 = *v46;
            if ( *v46 )
            {
              for ( j = v47 & v63; ; j = v63 & (j + 1) )
              {
                v49 = &v31[4 * j];
                if ( !*v49 )
                  break;
              }
              *v49 = v47;
              *((_QWORD *)v49 + 1) = *((_QWORD *)v46 + 1);
            }
            v46 += 4;
          }
          while ( &v33[4 * v61 + 4] != v46 );
        }
      }
LABEL_39:
      *v30 = v31;
      *((_DWORD *)v30 + 2) = v29;
      sub_823A00(v33, 16LL * v74);
LABEL_11:
      *(_QWORD *)(a1 + 368) = 0;
      goto LABEL_12;
    }
    sub_644060(a1);
LABEL_12:
    sub_866420(1);
  }
  v40 = v75;
  *(_QWORD *)(a2 + 80) = v75;
  v41 = *(_BYTE *)(v40 + 140);
  if ( (*(_BYTE *)(a1 + 123) & 0x10) != 0 )
  {
    if ( v41 == 12 )
    {
      v42 = v40;
      do
      {
        v42 = *(_QWORD *)(v42 + 160);
        v41 = *(_BYTE *)(v42 + 140);
      }
      while ( v41 == 12 );
    }
    if ( v41 )
    {
      v57 = *(_QWORD *)(a1 + 288);
      v76[0] = v40;
      sub_624710(v57, &v75, v76, a1, 0);
      v40 = v75;
    }
  }
  else
  {
    if ( v41 == 12 )
    {
      v50 = v40;
      do
      {
        v50 = *(_QWORD *)(v50 + 160);
        v41 = *(_BYTE *)(v50 + 140);
      }
      while ( v41 == 12 );
    }
    if ( v41 )
    {
      *(_QWORD *)(v40 + 160) = sub_72B6D0(&unk_4F077C8, 0);
      v40 = v75;
      *(_BYTE *)(a1 + 125) |= 8u;
    }
  }
  result = word_4F06418;
  *(_QWORD *)(a1 + 288) = v40;
  if ( word_4F06418[0] == 294 && (*(_BYTE *)(a1 + 122) & 0x10) == 0 )
  {
    if ( *(char *)(a1 + 133) < 0 )
      sub_6851C0(3217, &dword_4F063F8);
    return (unsigned __int16 *)sub_623350(a1, (unsigned int *)a2, (__int64)v77);
  }
  return result;
}
