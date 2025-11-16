// Function: sub_C396A0
// Address: 0xc396a0
//
__int64 __fastcall sub_C396A0(__int64 a1, _DWORD *a2, char a3, bool *a4)
{
  _DWORD *v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // r15d
  unsigned int v8; // eax
  int v9; // r12d
  unsigned int v10; // r9d
  char v11; // al
  __int64 v12; // r15
  char v13; // dl
  char v14; // al
  unsigned int v15; // r12d
  void *v16; // rax
  char v17; // al
  char v19; // al
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rdi
  unsigned int v23; // r9d
  char v24; // al
  __int64 v25; // rax
  char v26; // al
  int v27; // eax
  int v28; // ecx
  int v29; // edx
  int v30; // eax
  __int64 *v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rax
  int v34; // eax
  unsigned int v35; // [rsp+Ch] [rbp-54h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  unsigned int v37; // [rsp+10h] [rbp-50h]
  unsigned int v38; // [rsp+10h] [rbp-50h]
  unsigned __int8 v40; // [rsp+1Eh] [rbp-42h]
  bool v41; // [rsp+1Fh] [rbp-41h]
  int v43; // [rsp+28h] [rbp-38h]
  unsigned int v44; // [rsp+28h] [rbp-38h]
  unsigned int v45; // [rsp+28h] [rbp-38h]
  unsigned int v46; // [rsp+28h] [rbp-38h]
  unsigned int v47; // [rsp+2Ch] [rbp-34h]

  v5 = *(_DWORD **)a1;
  v40 = sub_C35FD0((_BYTE *)a1);
  v6 = 1;
  v7 = (unsigned int)(a2[2] + 64) >> 6;
  if ( v7 )
    v6 = (unsigned int)(a2[2] + 64) >> 6;
  v47 = v6;
  v8 = sub_C337D0(a1);
  v9 = a2[2] - v5[2];
  v10 = v8;
  v41 = a2 != (_DWORD *)&unk_3F655E0 && v5 == (_DWORD *)&unk_3F655E0;
  if ( v41 )
  {
    if ( (*(_BYTE *)(a1 + 20) & 7) == 1 )
    {
      v46 = v8;
      v31 = (__int64 *)sub_C33900(a1);
      v10 = v46;
      if ( *v31 < 0 )
      {
        v32 = (_QWORD *)sub_C33900(a1);
        v10 = v46;
        v41 = ((*v32 >> 62) ^ 1) & 1;
      }
    }
    else
    {
      v41 = 0;
    }
  }
  v43 = 0;
  if ( v9 >= 0 )
  {
LABEL_5:
    if ( v10 >= v47 )
      goto LABEL_6;
LABEL_45:
    v38 = v10;
    v12 = sub_2207820(8LL * v47);
    sub_C45D00(v12, 0, v47);
    v24 = *(_BYTE *)(a1 + 20) & 7;
    if ( v24 != 1 && (v24 == 3 || !v24) )
      goto LABEL_12;
    v25 = sub_C33900(a1);
    sub_C45D30(v12, v25, v38);
    sub_C33830(a1);
    *(_QWORD *)(a1 + 8) = v12;
    goto LABEL_13;
  }
  v19 = *(_BYTE *)(a1 + 20) & 7;
  if ( (*(_BYTE *)(a1 + 20) & 6) != 0 && v19 != 3 )
  {
    v45 = v10;
    v27 = sub_C34200(a1);
    v28 = *(_DWORD *)(a1 + 16);
    v29 = v27 + 1;
    v10 = v45;
    v30 = v27 + 1 - v5[2];
    if ( v28 + v30 < a2[1] )
      v30 = a2[1] - v28;
    if ( v30 < v9 )
      v30 = v9;
    if ( v30 >= 0 )
    {
      if ( -v9 < v29 )
      {
LABEL_67:
        v19 = *(_BYTE *)(a1 + 20) & 7;
        goto LABEL_37;
      }
      v30 = v29 + v9 - 1;
    }
    v9 -= v30;
    *(_DWORD *)(a1 + 16) = v28 + v30;
    if ( v9 >= 0 )
      goto LABEL_63;
    goto LABEL_67;
  }
LABEL_37:
  if ( v19 != 1 )
  {
    if ( v19 && v19 != 3 )
      goto LABEL_40;
LABEL_63:
    v43 = 0;
    goto LABEL_5;
  }
  v43 = 0;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) == 1 )
    goto LABEL_5;
LABEL_40:
  v44 = v10;
  v36 = sub_C33900(a1);
  v35 = v44;
  v20 = sub_C45DF0(v36, v44);
  v21 = (unsigned int)-v9;
  v43 = 0;
  v22 = v36;
  v23 = v35;
  if ( -v9 > v20 )
  {
    if ( (_DWORD)v21 == v20 + 1 )
    {
      v43 = 2;
    }
    else if ( (unsigned int)v21 > v35 << 6
           || (v34 = sub_C45D90(v36, (unsigned int)~v9), v43 = 3, v22 = v36, v21 = (unsigned int)-v9, v23 = v35, !v34) )
    {
      v43 = 1;
    }
  }
  v37 = v23;
  sub_C48220(v22, v23, v21);
  v10 = v37;
  if ( v37 < v47 )
    goto LABEL_45;
LABEL_6:
  if ( v7 <= 1 && v10 != 1 )
  {
    v11 = *(_BYTE *)(a1 + 20) & 7;
    if ( v11 == 1 || v11 && v11 != 3 )
    {
      v12 = *(_QWORD *)sub_C33900(a1);
LABEL_12:
      sub_C33830(a1);
      *(_QWORD *)(a1 + 8) = v12;
      goto LABEL_13;
    }
    sub_C33830(a1);
    *(_QWORD *)(a1 + 8) = 0;
  }
LABEL_13:
  *(_QWORD *)a1 = a2;
  if ( v9 > 0 )
  {
    v13 = *(_BYTE *)(a1 + 20);
    v14 = v13 & 7;
    if ( (v13 & 7) != 1 )
    {
      if ( v14 == 3 )
        goto LABEL_49;
      if ( !v14 )
      {
LABEL_17:
        if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
          goto LABEL_18;
        v15 = 16;
        sub_C36070(a1, 0, (v13 & 8) != 0, 0);
        *a4 = 1;
        goto LABEL_26;
      }
    }
    v16 = (void *)sub_C33900(a1);
    sub_C475D0(v16);
  }
  v13 = *(_BYTE *)(a1 + 20);
  v17 = v13 & 7;
  if ( (v13 & 7) == 1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) == 1 )
    {
      *a4 = v5[4] != 1;
      sub_C36070(a1, 0, (*(_BYTE *)(a1 + 20) & 8) != 0, 0);
      return v40;
    }
    if ( v5[5] == 2 && *(_DWORD *)(*(_QWORD *)a1 + 20LL) != 2 )
      sub_C36070(a1, 0, 0, 0);
    *a4 = v41 || v43 != 0;
    if ( !v41 && *(_UNKNOWN **)a1 == &unk_3F655E0 )
    {
      v33 = sub_C33900(a1);
      sub_C45DB0(v33, 63);
    }
    v15 = 0;
    if ( v40 )
    {
      v15 = 1;
      sub_C39170(a1);
    }
    goto LABEL_26;
  }
  if ( !v17 )
    goto LABEL_17;
  if ( v17 != 3 )
  {
    v15 = sub_C36450(a1, a3, v43);
    *a4 = v15 != 0;
    goto LABEL_26;
  }
LABEL_49:
  if ( *(_DWORD *)(*(_QWORD *)a1 + 20LL) != 2 )
  {
LABEL_18:
    v15 = 0;
    *a4 = 0;
    if ( (*(_BYTE *)(a1 + 20) & 7) != 3 )
      return v15;
    goto LABEL_19;
  }
  if ( v5[5] == 2 )
  {
    v26 = 0;
    v15 = 0;
  }
  else
  {
    v26 = (v13 & 8) != 0;
    v15 = -v26 & 0x10;
  }
  *a4 = v26;
  *(_BYTE *)(a1 + 20) &= ~8u;
LABEL_26:
  if ( (*(_BYTE *)(a1 + 20) & 7) != 3 )
    return v15;
LABEL_19:
  if ( !*(_BYTE *)(*(_QWORD *)a1 + 24LL) )
    sub_C35A40(a1, 0);
  return v15;
}
