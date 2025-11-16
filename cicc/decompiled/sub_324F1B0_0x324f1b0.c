// Function: sub_324F1B0
// Address: 0x324f1b0
//
__int64 __fastcall sub_324F1B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int16 v6; // ax
  __int64 v7; // r13
  unsigned __int8 v8; // al
  __int64 v9; // rdi
  const void *v10; // rax
  size_t v11; // rdx
  __int64 v12; // rdx
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned __int64 **v17; // r15
  __int64 v18; // r8
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // rdi
  unsigned __int8 v22; // al
  __int64 v23; // r14
  unsigned __int8 *v24; // rsi
  unsigned __int8 *v25; // r14
  unsigned __int64 **v26; // r8
  unsigned int v28; // r15d
  unsigned __int16 v29; // ax
  __int64 v30; // rax
  unsigned __int64 **v31; // r15
  __int64 v32; // rcx
  __int16 v33; // dx
  unsigned int v34; // r15d
  unsigned int v35; // eax
  __int64 v36; // rax
  unsigned __int64 v37; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+20h] [rbp-50h]
  unsigned __int64 v40; // [rsp+28h] [rbp-48h]
  unsigned __int64 v41; // [rsp+28h] [rbp-48h]
  __int64 v42[8]; // [rsp+30h] [rbp-40h] BYREF

  v5 = a3 - 16;
  v6 = sub_AF18C0(a3);
  v7 = sub_324C6D0(a1, v6, a2, 0);
  v8 = *(_BYTE *)(a3 - 16);
  if ( (v8 & 2) != 0 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 16LL);
    if ( !v9 )
      goto LABEL_21;
  }
  else
  {
    v9 = *(_QWORD *)(v5 - 8LL * ((v8 >> 2) & 0xF) + 16);
    if ( !v9 )
    {
LABEL_6:
      v12 = v5 - 8LL * ((v8 >> 2) & 0xF);
      goto LABEL_7;
    }
  }
  v10 = (const void *)sub_B91420(v9);
  if ( v11 )
    sub_324AD70(a1, v7, 3, v10, v11);
  v8 = *(_BYTE *)(a3 - 16);
  if ( (v8 & 2) == 0 )
    goto LABEL_6;
LABEL_21:
  v12 = *(_QWORD *)(a3 - 32);
LABEL_7:
  sub_324CC60(a1, v7, *(_QWORD *)(v12 + 40));
  v13 = *(_BYTE *)(a3 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(a3 - 32);
  else
    v14 = v5 - 8LL * ((v13 >> 2) & 0xF);
  v15 = *(_QWORD *)(v14 + 24);
  if ( v15 )
    sub_32495E0(a1, v7, v15, 73);
  sub_3249E10(a1, v7, a3);
  if ( (unsigned __int16)sub_AF18C0(a3) == 28 && (*(_BYTE *)(a3 + 20) & 0x20) != 0 )
  {
    v36 = sub_A777F0(0x10u, a1 + 11);
    v31 = (unsigned __int64 **)v36;
    if ( v36 )
    {
      *(_QWORD *)v36 = 0;
      *(_DWORD *)(v36 + 8) = 0;
    }
    sub_3249B00(a1, (unsigned __int64 **)v36, 11, 18);
    sub_3249B00(a1, v31, 11, 6);
    sub_3249B00(a1, v31, 11, 16);
    sub_3249B00(a1, v31, 15, *(_QWORD *)(a3 + 32));
    sub_3249B00(a1, v31, 11, 28);
    sub_3249B00(a1, v31, 11, 6);
    v32 = 34;
    v33 = 11;
    goto LABEL_50;
  }
  v39 = *(_QWORD *)(a3 + 24);
  v40 = sub_3212020(a3);
  v16 = sub_AF18D0(a3);
  if ( (*(_BYTE *)(a3 + 22) & 8) != 0 )
  {
    v17 = (unsigned __int64 **)(v7 + 8);
    if ( *(_BYTE *)(a1[26] + 3685) )
    {
      BYTE2(v42[0]) = 0;
      sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 11, v42[0], v40 >> 3);
    }
    BYTE2(v42[0]) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 13, v42[0], v39);
    v18 = *(_QWORD *)(a3 + 32);
    v19 = -(int)v40;
    if ( !*(_BYTE *)(a1[26] + 3685) )
    {
      BYTE2(v42[0]) = 0;
      v41 = (unsigned __int64)((unsigned int)v18 & v19) >> 3;
      sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 107, v42[0], v18);
LABEL_26:
      if ( (unsigned __int16)sub_3220AA0(a1[26]) > 2u )
      {
        v21 = a1[26];
        if ( !*(_BYTE *)(v21 + 3685) )
          goto LABEL_28;
        goto LABEL_43;
      }
      goto LABEL_47;
    }
    v37 = v40 + v18;
    v38 = ((_DWORD)v40 + (_DWORD)v18) & (unsigned int)v19;
    if ( *(_BYTE *)sub_31DA930(a1[23]) )
    {
      v20 = v37 - v38;
      if ( (__int64)(v37 - v38) >= 0 )
      {
LABEL_19:
        BYTE2(v42[0]) = 0;
        sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 12, v42[0], v20);
LABEL_20:
        v41 = (v38 - v40) >> 3;
        goto LABEL_26;
      }
    }
    else
    {
      v20 = v40 - v39 - v37 + v38;
      if ( v20 >= 0 )
        goto LABEL_19;
    }
    LODWORD(v42[0]) = 65549;
    sub_32498F0(a1, (unsigned __int64 **)(v7 + 8), 12, 65549, v20);
    goto LABEL_20;
  }
  v28 = v16 >> 3;
  v41 = *(_QWORD *)(a3 + 32) >> 3;
  v29 = sub_3220AA0(a1[26]);
  if ( v28 && v29 > 4u )
  {
    LODWORD(v42[0]) = 65551;
    sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 136, 65551, v28 & 0x1FFFFFFF);
  }
  if ( (unsigned __int16)sub_3220AA0(a1[26]) <= 2u )
  {
LABEL_47:
    v30 = sub_A777F0(0x10u, a1 + 11);
    v31 = (unsigned __int64 **)v30;
    if ( v30 )
    {
      *(_QWORD *)v30 = 0;
      *(_DWORD *)(v30 + 8) = 0;
    }
    sub_3249B00(a1, (unsigned __int64 **)v30, 11, 35);
    v32 = v41;
    v33 = 15;
LABEL_50:
    sub_3249B00(a1, v31, v33, v32);
    sub_3249620(a1, v7, 56, (__int64 **)v31);
    goto LABEL_28;
  }
  v21 = a1[26];
  v17 = (unsigned __int64 **)(v7 + 8);
LABEL_43:
  if ( (unsigned __int16)sub_3220AA0(v21) == 3 )
    LODWORD(v42[0]) = 65551;
  else
    BYTE2(v42[0]) = 0;
  sub_3249A20(a1, v17, 56, v42[0], v41);
LABEL_28:
  sub_3249F00(a1, v7, *(_DWORD *)(a3 + 20));
  if ( (*(_BYTE *)(a3 + 20) & 0x20) != 0 )
  {
    LODWORD(v42[0]) = 65547;
    sub_3249A20(a1, (unsigned __int64 **)(v7 + 8), 76, 65547, 1);
  }
  v22 = *(_BYTE *)(a3 - 16);
  if ( (v22 & 2) != 0 )
    v23 = *(_QWORD *)(a3 - 32);
  else
    v23 = v5 - 8LL * ((v22 >> 2) & 0xF);
  v24 = *(unsigned __int8 **)(v23 + 32);
  if ( v24 )
  {
    if ( *v24 == 28 )
    {
      v25 = sub_3247C80((__int64)a1, v24);
      if ( v25 )
      {
        v26 = (unsigned __int64 **)(v7 + 8);
        if ( (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
          || (v34 = (unsigned __int16)sub_3220AA0(a1[26]),
              v35 = sub_E06A90(16365),
              v26 = (unsigned __int64 **)(v7 + 8),
              v34 >= v35) )
        {
          v42[1] = (__int64)v25;
          v42[0] = 0x133FED00000007LL;
          sub_3248F80(v26, a1 + 11, v42);
        }
      }
    }
  }
  if ( (*(_BYTE *)(a3 + 20) & 0x40) != 0 )
    sub_3249FA0(a1, v7, 52);
  return v7;
}
