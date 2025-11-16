// Function: sub_824860
// Address: 0x824860
//
__int64 __fastcall sub_824860(__int64 a1, unsigned __int8 a2)
{
  unsigned int v2; // r13d
  char *v4; // r12
  int v6; // r14d
  int v7; // r15d
  unsigned __int8 *v8; // rax
  char v9; // dl
  __int64 v10; // rdi
  __int64 v11; // rax
  const char *v12; // r12
  __int64 v13; // rdi
  __int64 v14; // r13
  int v15; // eax
  int v16; // r14d
  __int64 v17; // rcx
  unsigned int i; // r15d
  int v19; // eax
  const char **v20; // r13
  const char *v21; // rdi
  const char *v22; // r13
  const char *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  const char *v28; // r14
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  _QWORD *v32; // rdx
  __int64 v33; // rax
  FILE *v34; // rax
  FILE *v35; // r12
  char v36; // r13
  const char *v37; // r14
  __int64 v38; // r13
  int v39; // eax
  int v40; // ecx
  __int64 v41; // rdx
  unsigned int j; // r15d
  int v43; // eax
  const char **v44; // r13
  const char *v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  _QWORD *v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // [rsp+8h] [rbp-58h]
  int v55; // [rsp+8h] [rbp-58h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  unsigned __int8 v57; // [rsp+27h] [rbp-39h] BYREF
  int v58[14]; // [rsp+28h] [rbp-38h] BYREF

  v2 = 1;
  if ( !*(_QWORD *)(a1 + 24) )
  {
    v2 = unk_4D047F4;
    if ( unk_4D047F4 )
    {
      return 0;
    }
    else
    {
      v4 = *(char **)(a1 + 8);
      if ( *(_BYTE *)a1 != 1 )
      {
        v13 = *(_QWORD *)(a1 + 8);
        v57 = a2;
        v14 = qword_4D048F0;
        v15 = sub_887620(v13);
        v16 = *(_DWORD *)(v14 + 8);
        v17 = *(_QWORD *)v14;
        for ( i = v16 & v15; ; i = v16 & (i + 1) )
        {
          v20 = (const char **)(v17 + 16LL * i);
          v21 = *v20;
          if ( *v20 && v4 )
          {
            v54 = v17;
            v19 = strcmp(v21, v4);
            v17 = v54;
            if ( !v19 )
              goto LABEL_18;
          }
          else if ( v21 == v4 )
          {
LABEL_18:
            v22 = v20[1];
            if ( v22 )
              goto LABEL_19;
LABEL_28:
            v32 = qword_4D048E8;
            v33 = qword_4D048E8[2];
            if ( !v33 )
              goto LABEL_20;
            do
            {
              v34 = sub_7244D0(*(char **)(*v32 + 8 * v33 - 8), "rb", v58);
              v35 = v34;
              if ( v34 )
              {
                v36 = sub_8241D0(v34);
                fclose(v35);
                if ( (unsigned __int8)(v36 - 1) <= 2u )
                  goto LABEL_51;
              }
              v32 = qword_4D048E8;
              v33 = qword_4D048E8[2] - 1LL;
              qword_4D048E8[2] = v33;
            }
            while ( v33 );
            v37 = *(const char **)(a1 + 8);
            v38 = qword_4D048F0;
            v39 = sub_887620(v37);
            v40 = *(_DWORD *)(v38 + 8);
            v41 = *(_QWORD *)v38;
            for ( j = v40 & v39; ; j = v40 & (j + 1) )
            {
              v44 = (const char **)(v41 + 16LL * j);
              v45 = *v44;
              if ( *v44 && v37 )
              {
                v55 = v40;
                v56 = v41;
                v43 = strcmp(v45, v37);
                v41 = v56;
                v40 = v55;
                if ( !v43 )
                  goto LABEL_42;
              }
              else
              {
                if ( v37 == v45 )
                {
LABEL_42:
                  v22 = v44[1];
                  if ( v22 )
                  {
LABEL_19:
                    if ( !(unsigned int)sub_824470(&v57, (__int64)v22) )
                      goto LABEL_20;
                    *(_BYTE *)a1 = v57;
                    *(_QWORD *)(a1 + 24) = sub_724840(dword_4F073B8[0], v22);
LABEL_45:
                    if ( *(_BYTE *)a1 > 3u )
                      sub_685200(0xC4Bu, dword_4F07508, *(_QWORD *)(a1 + 8));
LABEL_51:
                    sub_721090();
                  }
LABEL_20:
                  v23 = *(const char **)(a1 + 8);
                  v24 = qword_4F07698;
                  sub_823800(qword_4F1F638);
                  sub_824240(v23);
                  sub_8242F0(v23);
                  sub_8238B0(
                    (_QWORD *)qword_4F1F638,
                    *(const void **)(qword_4F1F630 + 32),
                    *(_QWORD *)(qword_4F1F630 + 16),
                    v25,
                    v26,
                    v27);
                  if ( *(_QWORD *)(qword_4F1F628 + 16) > 1u )
                  {
                    sub_823950(qword_4F1F638);
                    v50 = (_QWORD *)qword_4F1F638;
                    v51 = *(_QWORD *)(qword_4F1F638 + 16);
                    if ( (unsigned __int64)(v51 + 1) > *(_QWORD *)(qword_4F1F638 + 8) )
                    {
                      sub_823810((_QWORD *)qword_4F1F638, v51 + 1, v46, v47, v48, v49);
                      v50 = (_QWORD *)qword_4F1F638;
                      v51 = *(_QWORD *)(qword_4F1F638 + 16);
                    }
                    *(_BYTE *)(v50[4] + v51) = 45;
                    v52 = qword_4F1F628;
                    ++v50[2];
                    sub_8238B0(v50, *(const void **)(v52 + 32), *(_QWORD *)(v52 + 16), v47, v48, v49);
                  }
                  v28 = *(const char **)(qword_4F1F638 + 32);
                  if ( !v24 )
LABEL_26:
                    sub_685200(0xC03u, dword_4F07508, *(_QWORD *)(a1 + 8));
                  while ( 1 )
                  {
                    sub_720CF0(*(char **)v24, v28, (_QWORD *)qword_4F1F640);
                    sub_823950(qword_4F1F640);
                    sub_8238B0((_QWORD *)qword_4F1F640, ".ext", 5u, v29, v30, v31);
                    LOBYTE(v58[0]) = 2;
                    if ( (unsigned __int8)(a2 - 2) <= 1u )
                    {
                      sub_722380("edgm", qword_4F1F640);
                      if ( sub_7212B0(*(_QWORD *)(qword_4F1F640 + 32)) )
                      {
                        if ( (unsigned int)sub_824470((unsigned __int8 *)v58, *(_QWORD *)(qword_4F1F640 + 32)) )
                          break;
                      }
                    }
                    v24 = *(_QWORD *)(v24 + 16);
                    if ( !v24 )
                      goto LABEL_26;
                  }
                  v53 = qword_4F1F640;
                  *(_BYTE *)a1 = 2;
                  *(_QWORD *)(a1 + 24) = sub_724840(dword_4F073B8[0], *(const char **)(v53 + 32));
                  goto LABEL_45;
                }
                if ( !v45 )
                  goto LABEL_20;
              }
            }
          }
          if ( !v21 )
            goto LABEL_28;
        }
      }
      v6 = dword_4F07594;
      dword_4F07594 = 1;
      v7 = dword_4F07598;
      dword_4F07598 = 1;
      if ( unk_4D04444 && sub_7215C0((unsigned __int8 *)v4) )
        v4 = sub_721630(v4);
      v8 = (unsigned __int8 *)sub_7B06F0(v4, *(_BYTE *)(a1 + 40) & 1, 0, 0);
      if ( !v8 )
        sub_685200(0xC49u, dword_4F07508, *(_QWORD *)(a1 + 8));
      v9 = *(_BYTE *)(a1 + 40);
      v10 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 16) = v8;
      LOBYTE(v58[0]) = a2;
      v11 = sub_7B07A0(v10, v8, v9 & 1);
      v12 = (const char *)v11;
      if ( v11 && (unsigned int)sub_824470((unsigned __int8 *)v58, v11) )
      {
        v2 = 1;
        *(_BYTE *)a1 = v58[0];
        *(_QWORD *)(a1 + 24) = sub_724840(dword_4F073B8[0], v12);
      }
      dword_4F07598 = v7;
      dword_4F07594 = v6;
    }
  }
  return v2;
}
