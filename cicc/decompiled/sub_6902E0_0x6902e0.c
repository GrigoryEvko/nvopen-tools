// Function: sub_6902E0
// Address: 0x6902e0
//
_QWORD *__fastcall sub_6902E0(__int64 a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  int *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 **v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  __int64 i; // r14
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  char v34; // al
  bool v35; // zf
  int v36; // eax
  __int64 v37; // [rsp+8h] [rbp-528h]
  int v38; // [rsp+20h] [rbp-510h] BYREF
  unsigned int v39; // [rsp+24h] [rbp-50Ch] BYREF
  __int64 v40; // [rsp+28h] [rbp-508h] BYREF
  __int64 v41; // [rsp+30h] [rbp-500h] BYREF
  __int64 *v42; // [rsp+38h] [rbp-4F8h] BYREF
  _BYTE v43[160]; // [rsp+40h] [rbp-4F0h] BYREF
  _QWORD v44[44]; // [rsp+E0h] [rbp-450h] BYREF
  _QWORD v45[44]; // [rsp+240h] [rbp-2F0h] BYREF
  __int64 v46[50]; // [rsp+3A0h] [rbp-190h] BYREF

  v6 = a1;
  v7 = *(_QWORD *)a2;
  v38 = 0;
  v42 = 0;
  v8 = qword_4D03C58;
  v37 = qword_4D03C58;
  if ( (*(_BYTE *)(*(_QWORD *)(v7 + 96) + 178LL) & 0x20) != 0
    || (v9 = a2, (*(_BYTE *)(*((_QWORD *)a2 + 21) + 110LL) & 4) != 0) )
  {
    v38 = 32;
LABEL_4:
    v10 = sub_72CBE0(a1, a2, a3, v8, a5, a6);
    *(_BYTE *)(v6 + 193) |= 0x22u;
    *(_BYTE *)(v6 + 206) |= 0x10u;
    v13 = v10;
    goto LABEL_5;
  }
  sub_7296C0(&v39);
  sub_6E1DD0(&v41);
  a2 = (int *)v43;
  sub_6E1E00(5, v43, 0, 1);
  *(_DWORD *)(unk_4D03C50 + 18LL) |= 0x11080u;
  qword_4D03C58 = &v42;
  v40 = sub_724DC0(5, v43, v15, &qword_4D03C58, v16, v17);
  v18 = (__int64 **)**((_QWORD **)v9 + 21);
  if ( v18 )
  {
    while ( 1 )
    {
      if ( ((_BYTE)v18[12] & 1) != 0 )
      {
        v19 = sub_73C570(v18[5], 1, -1);
        v20 = sub_72D2E0(v19, 0);
        sub_72BB40(v20, v40);
        v21 = sub_730690(v40);
        v22 = sub_73DCD0(v21);
        sub_6E7150(v22, v44);
        v23 = sub_730690(v40);
        v24 = sub_73DCD0(v23);
        sub_6E7150(v24, v45);
        sub_68FEF0(v44, v45, &dword_4F063F8, dword_4F06650[0], 0, (__int64)v46);
        a2 = &v38;
        sub_68B310(v46[0], &v38);
        sub_6E4710(v46);
        v25 = v38;
        if ( (*(_BYTE *)(unk_4D03C50 + 19LL) & 1) != 0 || (v38 & 0x20) != 0 )
          break;
      }
      v18 = (__int64 **)*v18;
      if ( !v18 )
        goto LABEL_17;
    }
LABEL_28:
    v38 = v25 | 0x20;
  }
  else
  {
LABEL_17:
    for ( i = **(_QWORD **)(*(_QWORD *)v9 + 96LL); i; i = *(_QWORD *)(i + 16) )
    {
      if ( *(_BYTE *)(i + 80) == 8 )
      {
        v27 = *(_QWORD *)(*(_QWORD *)(i + 88) + 120LL);
        if ( (unsigned int)sub_8D3410(v27) )
          v27 = sub_8D40F0(v27);
        if ( (unsigned int)sub_8D3A70(v27) || (unsigned int)sub_8D2870(v27) )
        {
          v28 = sub_73C570(v27, 1, -1);
          v29 = sub_72D2E0(v28, 0);
          sub_72BB40(v29, v40);
          v30 = sub_730690(v40);
          v31 = sub_73DCD0(v30);
          sub_6E7150(v31, v44);
          v32 = sub_730690(v40);
          v33 = sub_73DCD0(v32);
          sub_6E7150(v33, v45);
          sub_68FEF0(v44, v45, &dword_4F063F8, dword_4F06650[0], 0, (__int64)v46);
          a2 = &v38;
          sub_68B310(v46[0], &v38);
          sub_6E4710(v46);
          v25 = v38;
        }
        else
        {
          while ( 1 )
          {
            v34 = *(_BYTE *)(v27 + 140);
            if ( v34 != 12 )
              break;
            v27 = *(_QWORD *)(v27 + 160);
          }
          if ( v34 == 3 )
          {
            v25 = v38 | 4;
            v38 |= 4u;
          }
          else
          {
            if ( v34 == 6 )
            {
              v35 = (unsigned int)sub_8D2340(v27) == 0;
              v36 = v38;
              if ( !v35 )
              {
LABEL_42:
                v25 = v36 | 0x20;
                goto LABEL_28;
              }
            }
            else
            {
              v35 = v34 == 2;
              v36 = v38;
              if ( !v35 )
                goto LABEL_42;
            }
            v25 = v36 | 1;
            v38 = v25;
          }
        }
        if ( (*(_BYTE *)(unk_4D03C50 + 19LL) & 1) != 0 || (v25 & 0x20) != 0 )
          goto LABEL_28;
      }
    }
  }
  sub_724E30(&v40);
  sub_6E2B30();
  sub_6E1DF0(v41);
  a1 = v39;
  sub_729730(v39);
  if ( (v38 & 0x20) != 0 )
    goto LABEL_4;
  if ( (v38 & 0x10) != 0 )
    goto LABEL_33;
  if ( (v38 & 8) != 0 )
  {
    if ( (v38 & 6) != 0 )
    {
LABEL_33:
      v13 = sub_72CD30();
      goto LABEL_5;
    }
    v13 = sub_72CD00();
  }
  else if ( (v38 & 4) != 0 )
  {
    v13 = sub_72CCD0();
  }
  else if ( (v38 & 2) != 0 )
  {
    v13 = sub_72CCA0();
  }
  else
  {
    v13 = sub_72CC70();
  }
LABEL_5:
  sub_68A860(v13, v6 + 64, v6, v11, v12);
  if ( (*(_BYTE *)(v6 + 206) & 0x10) == 0 )
  {
    if ( v42 )
    {
      if ( (*(_BYTE *)(v6 + 193) & 5) != 0 && (*(_BYTE *)(v6 + 195) & 3) != 1 )
        sub_6854C0(0xC1Cu, (FILE *)(v6 + 64), *v42);
      *(_BYTE *)(v6 + 193) &= ~2u;
    }
    else
    {
      *(_BYTE *)(v6 + 193) |= 2u;
    }
  }
  qword_4D03C58 = v37;
  return &qword_4D03C58;
}
