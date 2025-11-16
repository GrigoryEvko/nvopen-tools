// Function: sub_6AF3D0
// Address: 0x6af3d0
//
__int64 __fastcall sub_6AF3D0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int16 v5; // r15
  __int64 v6; // r14
  __int64 v7; // r12
  int v8; // ebx
  _QWORD *v9; // rdx
  int v10; // eax
  _QWORD *v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // rdi
  bool v18; // r14
  int v19; // eax
  __int64 v20; // r14
  __int64 v21; // r14
  int v22; // eax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // [rsp+8h] [rbp-268h]
  __int64 v33; // [rsp+10h] [rbp-260h]
  __int64 v34; // [rsp+10h] [rbp-260h]
  __int64 v35; // [rsp+10h] [rbp-260h]
  int v36; // [rsp+1Ch] [rbp-254h]
  _BOOL4 v37; // [rsp+2Ch] [rbp-244h] BYREF
  __int64 v38; // [rsp+30h] [rbp-240h] BYREF
  __int64 v39; // [rsp+38h] [rbp-238h] BYREF
  _BYTE v40[160]; // [rsp+40h] [rbp-230h] BYREF
  _QWORD v41[8]; // [rsp+E0h] [rbp-190h] BYREF
  char v42[8]; // [rsp+124h] [rbp-14Ch] BYREF
  int v43; // [rsp+12Ch] [rbp-144h]
  __int16 v44; // [rsp+130h] [rbp-140h]

  v6 = a1;
  v7 = a2;
  if ( a1 )
  {
    v8 = 0;
    a2 = (__int64)&v39;
    sub_6E4430(a1, &v39, v41, &v37);
    if ( qword_4D03C50 )
      v8 = *(_BYTE *)(qword_4D03C50 + 19LL) & 1;
    a4 = *(unsigned int *)(*(_QWORD *)a1 + 44LL);
    v5 = *(_WORD *)(*(_QWORD *)a1 + 48LL);
    v36 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
  }
  else
  {
    v8 = 0;
    v39 = *(_QWORD *)&dword_4F063F8;
  }
  if ( !dword_4D048B8 )
  {
    a5 = dword_4F077BC;
    if ( !dword_4F077BC || (a3 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0) )
    {
      if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, dword_4F077BC, dword_4D048B8) )
      {
        a2 = (__int64)&v39;
        a1 = 540;
        sub_6851C0(0x21Cu, &v39);
      }
      goto LABEL_10;
    }
  }
  if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, a5, dword_4D048B8) )
    {
      a2 = (__int64)&v39;
      a1 = 57;
      v8 = 1;
      sub_6851C0(0x39u, &v39);
      goto LABEL_11;
    }
LABEL_10:
    v8 = 1;
    goto LABEL_11;
  }
  a1 = (__int64)&v39;
  if ( (unsigned int)sub_6E9250(&v39) )
    goto LABEL_10;
  if ( dword_4F04C58 != -1
    && (v9 = qword_4F04C68, (*(_BYTE *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216) + 198LL) & 0x10) != 0) )
  {
    a2 = (__int64)&dword_4F063F8;
    a1 = 3517;
    v8 = 1;
    sub_6851C0(0xDBDu, &dword_4F063F8);
  }
  else
  {
    a1 = dword_4D04324;
    if ( dword_4D04324 )
    {
      a2 = 876;
      a1 = (__int64)&v39;
      sub_684AB0(&v39, 0x36Cu);
    }
  }
LABEL_11:
  if ( v6 )
  {
    if ( !v37 )
      goto LABEL_13;
    a2 = (__int64)v40;
    sub_6E2140(*(unsigned __int8 *)(qword_4D03C50 + 16LL), v40, 0, 0, v6);
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 4u;
  }
  else
  {
    v5 = WORD2(qword_4F063F0);
    v36 = qword_4F063F0;
    sub_7B8B50(a1, a2, v9, (unsigned int)qword_4F063F0);
    v22 = sub_692B20(word_4F06418[0]);
    v37 = v22 != 0;
    if ( !v22 )
      goto LABEL_13;
    sub_6E2140(*(unsigned __int8 *)(qword_4D03C50 + 16LL), v40, 0, 0, 0);
    a2 = 0;
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 4u;
    sub_69ED20((__int64)v41, 0, 2, 0);
    v5 = v44;
    v36 = v43;
  }
  v33 = v41[0];
  if ( (unsigned int)sub_8D3A70(v41[0]) )
  {
    v19 = dword_4F077C4;
    v20 = v41[0];
    goto LABEL_33;
  }
  a2 = 0;
  sub_6F69D0(v41, 0);
  v19 = dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    v26 = v41[0];
    v20 = v41[0];
    if ( unk_4F07778 > 201102 || (a2 = dword_4F07774) != 0 )
    {
      a2 = dword_4F077BC;
      if ( dword_4F077BC )
        goto LABEL_62;
LABEL_58:
      v33 = v26;
      v20 = v26;
LABEL_33:
      if ( v19 != 2 )
        goto LABEL_34;
LABEL_62:
      if ( (unsigned int)sub_8D23B0(v20) )
        sub_8AE000(v20);
      goto LABEL_34;
    }
    if ( dword_4D04964 )
      goto LABEL_62;
LABEL_57:
    v20 = v26;
    if ( dword_4F077BC )
      goto LABEL_33;
    goto LABEL_58;
  }
  if ( !dword_4D04964 )
  {
    v26 = v41[0];
    goto LABEL_57;
  }
  v20 = v41[0];
LABEL_34:
  if ( (unsigned int)sub_8D2600(v20) )
  {
    a2 = (__int64)v41;
    sub_6E68E0(601, v41);
  }
  else if ( (unsigned int)sub_8D23B0(v33) )
  {
    a2 = v33;
    sub_6E5F60(v42, v33, 8);
    sub_6E6840(v41);
  }
  else if ( dword_4D047EC && (unsigned int)sub_8DD010(v20) )
  {
    a2 = (__int64)v41;
    sub_6E68E0(975, v41);
  }
  else if ( (unsigned int)sub_8D2E30(v20) )
  {
    v21 = sub_8D46C0(v20);
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v21) )
      sub_8AE000(v21);
    if ( (unsigned int)sub_8D23B0(v21) && !(unsigned int)sub_8D2600(v21) )
    {
      a2 = (__int64)v41;
      sub_6E68E0(1272, v41);
    }
  }
  else if ( (unsigned int)sub_8D5830(v20) )
  {
    if ( (unsigned int)sub_6E5430(v20, a2, v28, v29, v30, v31) )
    {
      a2 = 322;
      sub_5EB950(8u, 322, v20, (__int64)v42);
    }
    sub_6E6840(v41);
  }
LABEL_13:
  if ( v8 )
  {
    sub_6E6260(v7);
    if ( !v37 )
      goto LABEL_15;
    v17 = v41;
    sub_6E6450(v41);
  }
  else
  {
    v12 = (_QWORD *)sub_726700(8);
    *v12 = sub_72CBE0(8, a2, v13, v14, v15, v16);
    if ( v37 )
    {
      v23 = v41[0];
      if ( (unsigned int)sub_8D3A70(v41[0]) )
      {
        sub_8470D0((unsigned int)v41, v23, 0, 128, 144, 0, (__int64)&v38);
        a2 = v23;
        v32 = v12[7];
        v34 = v38;
        v24 = sub_6EB2F0(v23, v23, v42, 0);
        v25 = v38;
        *(_QWORD *)(v32 + 16) = v24;
        *(_QWORD *)(v12[7] + 8LL) = v25;
        *(_QWORD *)v12[7] = v23;
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
          sub_8DCE90(v23);
        if ( !v34 )
        {
          v17 = (_QWORD *)v7;
          sub_6E6260(v7);
          goto LABEL_26;
        }
      }
      else
      {
        v35 = sub_6F6F40(v41, 0);
        v27 = sub_6EB460(3, v23, v42);
        v38 = v27;
        *(_QWORD *)(v27 + 56) = v35;
        *(_QWORD *)(v12[7] + 8LL) = v27;
        *(_QWORD *)v12[7] = v23;
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
          sub_8DCE90(v23);
      }
    }
    else
    {
      v12[7] = 0;
    }
    sub_6E70E0(v12, v7);
    a2 = (__int64)&v39;
    v17 = v12;
    sub_6E3AC0(v12, &v39, 0, 0);
  }
LABEL_26:
  if ( v37 )
  {
    v18 = (*(_BYTE *)(qword_4D03C50 + 20LL) & 4) != 0;
    sub_6891A0();
    sub_6E2B30(v17, a2);
    if ( v18 )
      *(_BYTE *)(qword_4D03C50 + 20LL) |= 4u;
  }
LABEL_15:
  v10 = v39;
  *(_WORD *)(v7 + 80) = v5;
  *(_DWORD *)(v7 + 68) = v10;
  *(_WORD *)(v7 + 72) = WORD2(v39);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v7 + 68);
  *(_DWORD *)(v7 + 76) = v36;
  unk_4F061D8 = *(_QWORD *)(v7 + 76);
  sub_6E3280(v7, &v39);
  return sub_6E26D0(2, v7);
}
