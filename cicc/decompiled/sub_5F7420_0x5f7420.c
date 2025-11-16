// Function: sub_5F7420
// Address: 0x5f7420
//
__int64 **__fastcall sub_5F7420(__int64 *a1, _QWORD *a2, _DWORD *a3, int a4)
{
  _QWORD *v4; // r15
  unsigned __int64 v7; // r12
  int v8; // eax
  unsigned __int64 v9; // rcx
  unsigned __int64 i; // rdx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rbx
  __int64 **v14; // r10
  char v16; // al
  bool v17; // r8
  int v18; // eax
  __int64 v19; // rdi
  int v20; // r11d
  char v21; // al
  __int64 v22; // r9
  int v23; // r11d
  _QWORD *v24; // r9
  int v25; // r13d
  __int64 v26; // r15
  __int64 **v27; // rbx
  char v28; // r14
  __int64 v29; // rax
  int v30; // eax
  int v31; // r14d
  _DWORD *v32; // [rsp+0h] [rbp-60h]
  int v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  bool v35; // [rsp+18h] [rbp-48h]
  _QWORD *v36; // [rsp+18h] [rbp-48h]
  __int64 **v37; // [rsp+18h] [rbp-48h]
  unsigned int v38; // [rsp+28h] [rbp-38h] BYREF
  int v39[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v4 = a2;
  v7 = (unsigned __int64)a1;
  v8 = sub_85E8D0();
  if ( v8 == -1 )
  {
    v13 = 0;
  }
  else
  {
    v9 = *(_QWORD *)(qword_4F04C68[0] + 776LL * v8 + 216);
    for ( i = v9 >> 3; ; LODWORD(i) = v11 + 1 )
    {
      v11 = qword_4F04C10[1] & i;
      v12 = (__int64 *)(*qword_4F04C10 + 16LL * v11);
      v13 = *v12;
      if ( v9 == *v12 )
        break;
      if ( !v13 )
        goto LABEL_7;
    }
    v13 = v12[1];
  }
LABEL_7:
  if ( a3 )
    *a3 = 0;
  v14 = sub_5EA830((__int64 ***)v13, a1, 0);
  if ( v14 )
    return v14;
  v16 = *(_BYTE *)(v13 + 24);
  v38 = 0;
  v17 = (v16 & 0x20) != 0;
  if ( a1 )
  {
    v34 = *(_QWORD *)(unk_4F04C50 + 32LL);
    v35 = (v16 & 0x20) != 0;
    v18 = sub_693860(a1, 1, v35, &v38);
    v19 = v38;
    v17 = v35;
    v20 = v18;
    v21 = a4 == 0;
    v22 = v34;
    v14 = 0;
    if ( !v20 )
    {
LABEL_20:
      v21 &= (_DWORD)v19 != 0;
      goto LABEL_39;
    }
    if ( v38 && !a4 )
    {
      sub_684B30(v38, a2);
      v38 = 0;
      v17 = v35;
      v22 = v34;
      v21 = 1;
      v14 = 0;
    }
    if ( (*(_BYTE *)(v13 + 24) & 0x10) == 0 )
    {
      if ( (*(_BYTE *)(v7 + 176) & 2) != 0 && a3 )
      {
LABEL_19:
        *a3 = 1;
        v19 = v38;
        goto LABEL_20;
      }
      v19 = (*(_BYTE *)(v7 + 172) & 1) == 0 ? 1735 : 1738;
LABEL_47:
      v38 = v19;
      goto LABEL_39;
    }
    if ( (*(_BYTE *)(v22 + 195) & 9) == 1 )
    {
      if ( (*(_BYTE *)(v7 + 176) & 2) != 0 && a3 )
        goto LABEL_19;
      if ( a4 )
        return v14;
      sub_6854C0(3202, a2, *(_QWORD *)v7);
      v14 = 0;
LABEL_52:
      v19 = v38;
      if ( v38 )
        goto LABEL_40;
      return v14;
    }
    if ( (*(_QWORD *)(v7 + 168) & 0x4000000000008000LL) == 0x4000000000008000LL )
    {
      v23 = *(_DWORD *)(*(_QWORD *)(v7 + 128) + 36LL);
      goto LABEL_24;
    }
  }
  else
  {
    if ( (v16 & 0x10) == 0 )
    {
      v19 = 1735;
      v21 = a4 == 0;
      goto LABEL_47;
    }
    if ( (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 195LL) & 9) == 1 )
    {
      if ( a4 )
        return v14;
      sub_6851C0(3203, a2);
      v14 = 0;
      goto LABEL_52;
    }
  }
  v23 = 0;
LABEL_24:
  v32 = a3;
  v24 = a2;
  v25 = v23;
  v26 = v13;
  v27 = v14;
  v33 = a4;
  v28 = v17;
  while ( 1 )
  {
    v36 = v24;
    v29 = sub_5F72B0(v26, v7, 0, 1, v28, v24, v39);
    v24 = v36;
    if ( v39[0] )
    {
      if ( v7 )
        break;
    }
    if ( !v27 )
      v27 = (__int64 **)v29;
    if ( !v39[0] )
      goto LABEL_34;
    if ( !v7 )
    {
LABEL_32:
      v30 = 1877;
      goto LABEL_33;
    }
LABEL_31:
    v30 = 1738;
    if ( (*(_BYTE *)(v7 + 172) & 1) == 0 )
      goto LABEL_32;
LABEL_33:
    v38 = v30;
LABEL_34:
    if ( v25 )
    {
      v7 = *(_QWORD *)(v7 + 112);
      if ( v7 )
      {
        if ( *(_DWORD *)(*(_QWORD *)(v7 + 128) + 36LL) == v25 )
          continue;
      }
    }
    v31 = v33;
    v14 = v27;
    v4 = v36;
    goto LABEL_38;
  }
  if ( (*(_BYTE *)(v7 + 176) & 2) == 0 || !v32 )
  {
    if ( !v27 )
      v27 = (__int64 **)v29;
    goto LABEL_31;
  }
  v31 = v33;
  v14 = v27;
  v4 = v36;
  *v32 = 1;
LABEL_38:
  v19 = v38;
  v21 = v31 == 0 && v38 != 0;
LABEL_39:
  if ( v21 )
  {
LABEL_40:
    v37 = v14;
    sub_6851C0(v19, v4);
    return v37;
  }
  return v14;
}
