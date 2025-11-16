// Function: sub_7D1970
// Address: 0x7d1970
//
__int64 __fastcall sub_7D1970(int a1, int a2, __int64 *a3, _DWORD *a4)
{
  int v4; // r8d
  _DWORD *v6; // r14
  __int64 v7; // r13
  int v8; // r10d
  int v9; // r9d
  bool v10; // dl
  int v11; // r9d
  __int64 result; // rax
  _BOOL4 v13; // edx
  __int64 v14; // rcx
  __int64 v15; // rdi
  char v16; // al
  __int64 v17; // rdi
  __int64 v18; // rcx
  char v19; // dl
  int v20; // r11d
  __int64 v21; // rax
  __int64 v22; // rbx
  _DWORD *v23; // rdx
  int *v24; // r14
  __int64 v25; // r13
  __int64 v26; // r12
  int v27; // ecx
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rcx
  char v31; // si
  char v32; // al
  _BOOL4 v33; // eax
  char v34; // al
  char v35; // si
  int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rdx
  __int64 *v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rdi
  int v44; // [rsp+0h] [rbp-70h]
  _DWORD *v45; // [rsp+0h] [rbp-70h]
  int v46; // [rsp+Ch] [rbp-64h]
  int v47; // [rsp+Ch] [rbp-64h]
  int v48; // [rsp+10h] [rbp-60h]
  int v49; // [rsp+10h] [rbp-60h]
  int v50; // [rsp+14h] [rbp-5Ch]
  int v51; // [rsp+14h] [rbp-5Ch]
  int v52; // [rsp+18h] [rbp-58h]
  _DWORD *v53; // [rsp+18h] [rbp-58h]
  int v54; // [rsp+18h] [rbp-58h]
  int v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  char v58; // [rsp+20h] [rbp-50h]
  int v59; // [rsp+28h] [rbp-48h]
  int v60; // [rsp+28h] [rbp-48h]
  __int64 v61; // [rsp+28h] [rbp-48h]
  int v62; // [rsp+28h] [rbp-48h]
  int v63; // [rsp+30h] [rbp-40h]
  __int64 v64; // [rsp+30h] [rbp-40h]
  __int64 v65; // [rsp+30h] [rbp-40h]
  int v66; // [rsp+30h] [rbp-40h]
  int v67; // [rsp+38h] [rbp-38h]
  int v68; // [rsp+38h] [rbp-38h]
  __int64 v69; // [rsp+38h] [rbp-38h]

  v4 = a1;
  v6 = a4;
  v7 = 0;
  if ( a2 != -1 )
    v7 = qword_4F04C68[0] + 776LL * a2;
  v8 = a4[34];
  if ( HIDWORD(qword_4D0495C) )
  {
    if ( (_BYTE)a1 != 7 )
      goto LABEL_5;
    if ( a4[15] )
    {
      a4[15] = 0;
      v11 = 0;
      result = 0;
      goto LABEL_19;
    }
  }
  else if ( (_BYTE)a1 != 7 )
  {
    goto LABEL_5;
  }
  if ( a4[14] )
  {
    v11 = 0;
    result = 0;
    goto LABEL_19;
  }
LABEL_5:
  if ( (*(_BYTE *)(v7 + 8) & 4) != 0 )
  {
    v67 = a4[34];
    result = sub_7D1590(a1, a2, a3, (__int64)a4);
    v8 = v67;
    v11 = 0;
    goto LABEL_19;
  }
  v9 = a4[13];
  if ( v9 )
  {
    v10 = 1;
    v11 = 0;
    result = 0;
    goto LABEL_8;
  }
  v15 = *(_QWORD *)(v7 + 184);
  if ( unk_4D03F98 )
  {
    if ( v15 )
    {
      if ( *(_QWORD *)(*a3 + 64) )
      {
        v16 = *(_BYTE *)(v15 + 28);
        if ( v16 == 3 || !v16 )
        {
          v59 = v4;
          v63 = a4[34];
          sub_824D70(v15);
          v4 = v59;
          v8 = v63;
          v9 = 0;
        }
      }
    }
  }
  if ( a2 == -1 )
    BUG();
  v7 = qword_4F04C68[0] + 776LL * a2;
  v17 = *(_QWORD *)(v7 + 24);
  if ( !v17 )
    v17 = v7 + 32;
  v18 = *(_QWORD *)(v17 + 136);
  if ( (*(_BYTE *)(v7 + 10) & 4) == 0 )
  {
    v20 = 0;
    if ( v18 )
      goto LABEL_36;
    v19 = *(_BYTE *)(v7 + 4);
LABEL_65:
    if ( (unsigned __int8)(v19 - 8) <= 1u )
    {
      v42 = **(_QWORD ***)(v7 + 408);
      if ( v42 )
      {
        while ( 1 )
        {
          v22 = v42[1];
          if ( *(_QWORD *)v22 == *a3 )
            break;
          v42 = (_QWORD *)*v42;
          if ( !v42 )
            goto LABEL_106;
        }
        v6[34] = 0;
        v20 = 0;
        v9 = 1;
        goto LABEL_38;
      }
LABEL_106:
      v69 = 0;
      goto LABEL_82;
    }
    v20 = 0;
    v22 = *(_QWORD *)(*a3 + 32);
    goto LABEL_37;
  }
  v19 = *(_BYTE *)(v7 + 4);
  if ( ((v19 - 15) & 0xFD) != 0 && v19 != 2 )
  {
    if ( v18 )
    {
      v20 = 0;
LABEL_36:
      v52 = v4;
      v55 = v20;
      v60 = v8;
      v64 = *(_QWORD *)(v17 + 136);
      v68 = v9;
      v21 = sub_883800(v17, *a3);
      v9 = v68;
      v18 = v64;
      v8 = v60;
      v20 = v55;
      v22 = v21;
      v4 = v52;
      goto LABEL_37;
    }
    goto LABEL_65;
  }
  v20 = 1;
  if ( v18 )
    goto LABEL_36;
  v37 = *(_QWORD *)(v7 + 184);
  v38 = 776LL * *(int *)(v37 + 240) + qword_4F04C68[0];
  if ( (*(_BYTE *)(v38 + 10) & 4) != 0 )
  {
    v22 = *(_QWORD *)(v37 + 248);
  }
  else
  {
    v40 = *(__int64 **)(v38 + 24);
    v41 = (__int64 *)(v38 + 32);
    if ( !v40 )
      v40 = v41;
    v22 = *v40;
  }
LABEL_37:
  if ( !v22 )
    goto LABEL_106;
LABEL_38:
  v23 = v6;
  v61 = 0;
  v24 = (int *)v7;
  v25 = v18;
  v69 = 0;
  v65 = 0;
  while ( 1 )
  {
    v27 = *v24;
    v28 = *(_DWORD *)(v22 + 40);
    if ( v25 )
    {
      v26 = *(_QWORD *)(v22 + 32);
      goto LABEL_40;
    }
    if ( !v20 )
      break;
    v26 = *(_QWORD *)(v22 + 16);
    if ( v27 != v28 )
      goto LABEL_41;
LABEL_46:
    if ( *(_QWORD *)v22 != *a3 )
      goto LABEL_41;
LABEL_47:
    v29 = *(unsigned __int8 *)(v22 + 80);
    v30 = v22;
    v31 = *(_BYTE *)(v22 + 80);
    if ( (_BYTE)v29 == 16 )
    {
      v30 = **(_QWORD **)(v22 + 88);
      v31 = *(_BYTE *)(v30 + 80);
    }
    if ( v31 == 24 )
      v30 = *(_QWORD *)(v30 + 88);
    if ( dword_4F04BA0[v29] != v23[31] )
      goto LABEL_41;
    v32 = *(_BYTE *)(v30 + 83);
    if ( (v32 & 0x40) != 0 || (*(_BYTE *)(v22 + 83) & 0x40) != 0 )
    {
      if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (unsigned __int64)(qword_4F077A8 - 50000LL) > 0x270F )
        goto LABEL_74;
      v35 = *(_BYTE *)(v30 + 80);
      if ( v35 == 20 )
        goto LABEL_54;
      if ( v35 != 17 )
        goto LABEL_74;
      v45 = v23;
      v47 = v4;
      v49 = v20;
      v51 = v8;
      v54 = v9;
      v57 = v30;
      v36 = sub_8780F0(v30);
      v30 = v57;
      v9 = v54;
      v8 = v51;
      v20 = v49;
      v4 = v47;
      v23 = v45;
      if ( !v36 )
      {
LABEL_74:
        if ( !v23[6] && !v23[8] )
          goto LABEL_41;
      }
      v32 = *(_BYTE *)(v30 + 83);
    }
LABEL_54:
    if ( v32 < 0 && unk_4F04C48 != -1 )
    {
      v39 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 368);
      if ( v39 )
      {
        if ( v30 == v39 )
          goto LABEL_41;
      }
    }
    v44 = v4;
    v46 = v20;
    v48 = v8;
    v50 = v9;
    v53 = v23;
    v56 = v30;
    v33 = sub_7CF0D0(v22, v30, v23);
    v23 = v53;
    v9 = v50;
    v8 = v48;
    v20 = v46;
    v4 = v44;
    if ( !v33 || (*(_BYTE *)(v22 + 82) & 1) != 0 )
      goto LABEL_41;
    if ( v53[1] )
    {
      if ( *(_BYTE *)(v22 + 80) != 3 )
        goto LABEL_110;
      if ( !v69 || (*(_BYTE *)(v69 + 80) & 0xF7) == 0x10 )
        v69 = v22;
LABEL_41:
      v22 = v26;
      goto LABEL_42;
    }
    v34 = *(_BYTE *)(v56 + 80);
    if ( (unsigned __int8)(v34 - 4) <= 2u || v34 == 3 && *(_BYTE *)(v56 + 104) )
    {
      if ( !v65 || (*(_BYTE *)(v65 + 80) & 0xF7) == 0x10 )
      {
        v65 = v22;
        v22 = v26;
        goto LABEL_42;
      }
      goto LABEL_41;
    }
    if ( *(_BYTE *)(v22 + 80) != 23 || !dword_4D044B8 )
    {
LABEL_110:
      v7 = (__int64)v24;
      v69 = v22;
      v6 = v53;
      goto LABEL_82;
    }
    v61 = v22;
    v22 = v26;
LABEL_42:
    if ( !v26 )
      goto LABEL_81;
  }
  if ( !v9 )
  {
    v26 = *(_QWORD *)(v22 + 8);
LABEL_40:
    if ( v27 != v28 )
      goto LABEL_41;
    goto LABEL_46;
  }
  if ( v27 == v28 && *a3 == *(_QWORD *)v22 )
  {
    v26 = 0;
    goto LABEL_47;
  }
LABEL_81:
  v7 = (__int64)v24;
  v6 = v23;
  if ( !v69 )
  {
    v43 = v61;
    if ( v65 )
      v43 = v65;
    v69 = v43;
  }
LABEL_82:
  v58 = v4;
  v62 = v8;
  v66 = v9;
  result = sub_7D1520(v7, v69, (__int64)a3, v6);
  v11 = v66;
  v8 = v62;
  LOBYTE(v4) = v58;
  v10 = result == 0;
LABEL_8:
  if ( (_BYTE)v4 == 7 && v10 )
  {
    v6[19] = 1;
    v13 = 1;
    if ( (dword_4F04C44 & unk_4F04C48) != -1 && !v6[23] )
    {
      v14 = *(_QWORD *)(v7 + 208);
      if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 9) <= 2u )
        v13 = (*(_DWORD *)(v14 + 176) & 0x11000) != 4096;
    }
    v6[20] = v13;
    result = 0;
    v6[24] = 0;
    *((_QWORD *)v6 + 14) = 0;
  }
LABEL_19:
  v6[34] = v8;
  v6[35] = v11;
  return result;
}
