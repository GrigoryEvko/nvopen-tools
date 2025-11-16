// Function: sub_798FD0
// Address: 0x798fd0
//
__int64 __fastcall sub_798FD0(__int64 a1, __int64 *a2, FILE *a3, _QWORD *a4, __int64 a5, char a6)
{
  __int64 v7; // r12
  __int64 *v8; // rax
  unsigned int v10; // r15d
  unsigned __int64 v11; // rax
  unsigned int v12; // r10d
  unsigned int v13; // ecx
  __int64 v14; // rsi
  unsigned int v15; // eax
  int *v16; // rdx
  int v17; // edi
  int v18; // eax
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // rsi
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // r10
  __int64 v27; // rax
  unsigned __int64 v28; // r11
  unsigned int v29; // r15d
  __int64 v30; // rbx
  unsigned int i; // eax
  __int64 v32; // r9
  unsigned __int64 *v33; // rdx
  int v34; // eax
  __int64 v35; // rax
  bool v36; // zf
  unsigned int v37; // eax
  __int64 v38; // r8
  unsigned __int64 v39; // r11
  __int64 *v40; // rbx
  __int64 v41; // r15
  __int64 v42; // r13
  __int64 *v43; // r12
  unsigned __int64 v44; // rcx
  unsigned int v45; // edx
  __int64 v46; // rax
  __int64 v47; // r11
  _QWORD *v48; // rbx
  __int64 v49; // rax
  int v50; // ecx
  __int64 v51; // rdi
  unsigned int v52; // edx
  __int64 v53; // rsi
  _QWORD *v54; // rax
  int v55; // esi
  __int64 v56; // rcx
  unsigned int v57; // edx
  _DWORD *j; // rax
  char v59; // al
  _DWORD *v61; // rdx
  int v62; // eax
  __int64 v63; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v64; // [rsp+8h] [rbp-C8h]
  __int64 v65; // [rsp+10h] [rbp-C0h]
  __int64 v66; // [rsp+18h] [rbp-B8h]
  unsigned int v67; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v68; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v69; // [rsp+20h] [rbp-B0h]
  __int64 v70; // [rsp+20h] [rbp-B0h]
  __int64 v71; // [rsp+28h] [rbp-A8h]
  __int64 v72; // [rsp+28h] [rbp-A8h]
  __int64 v73; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v74; // [rsp+28h] [rbp-A8h]
  __int64 v75; // [rsp+28h] [rbp-A8h]
  __int64 v76; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v77; // [rsp+30h] [rbp-A0h]
  __int64 v78; // [rsp+30h] [rbp-A0h]
  __int64 v79; // [rsp+30h] [rbp-A0h]
  unsigned int v80; // [rsp+38h] [rbp-98h]
  __int64 v81; // [rsp+38h] [rbp-98h]
  __int64 v83; // [rsp+40h] [rbp-90h]
  __int64 *v84; // [rsp+48h] [rbp-88h] BYREF
  int v85; // [rsp+5Ch] [rbp-74h] BYREF
  _QWORD v86[5]; // [rsp+60h] [rbp-70h] BYREF
  int v87; // [rsp+88h] [rbp-48h]
  int v88; // [rsp+8Ch] [rbp-44h]
  char v89; // [rsp+90h] [rbp-40h]

  v84 = a2;
  if ( !a2 )
    sub_721090();
  v7 = a1;
  v8 = a2;
  if ( (*((_BYTE *)a2 + 193) & 2) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      v10 = 0;
      sub_686E10(0xA8Fu, a3, *a2, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return v10;
    }
    return 0;
  }
  if ( (*((_BYTE *)a2 + 193) & 0x20) == 0 )
  {
    v81 = a5;
    sub_8AD0D0(*a2, 1, 8);
    v8 = v84;
    a5 = v81;
  }
  v10 = *((_DWORD *)v8 + 40);
  if ( !v10 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_686E10(0xA7Bu, a3, *v8, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      v8 = v84;
    }
    if ( (*((_BYTE *)v8 + 206) & 0x10) != 0 )
    {
      v59 = *(_BYTE *)(a1 + 132);
      if ( (v59 & 1) != 0 )
      {
        *(_BYTE *)(a1 + 132) = v59 | 0x40;
        return v10;
      }
    }
    return 0;
  }
  if ( (*((_BYTE *)v8 + 195) & 8) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      v10 = 0;
      sub_686E10(0xA7Cu, a3, *v8, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return v10;
    }
    return 0;
  }
  v11 = *(_QWORD *)(a1 + 120) + 1LL;
  *(_QWORD *)(a1 + 120) = v11;
  if ( v11 > qword_4D042E0 )
  {
    v10 = 0;
    sub_6855B0(0x97Fu, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
    return v10;
  }
  v12 = *(_DWORD *)(a1 + 128);
  v13 = *(_DWORD *)(a1 + 64);
  v85 = 0;
  v14 = *(_QWORD *)(a1 + 56);
  v80 = v12;
  *(_DWORD *)(a1 + 128) = v12 + 1;
  v15 = v13 & v12;
  v16 = (int *)(v14 + 4LL * (v13 & v12));
  v17 = *v16;
  if ( *v16 )
  {
    *v16 = v12;
    do
    {
      v15 = v13 & (v15 + 1);
      v61 = (_DWORD *)(v14 + 4LL * v15);
    }
    while ( *v61 );
    *v61 = v17;
  }
  else
  {
    *v16 = v12;
  }
  v18 = *(_DWORD *)(v7 + 68) + 1;
  *(_DWORD *)(v7 + 68) = v18;
  if ( 2 * v18 > v13 )
  {
    v78 = a5;
    sub_7702C0(v7 + 56);
    a5 = v78;
  }
  v19 = *(_QWORD *)(v7 + 16);
  v20 = qword_4F080A8;
  if ( (unsigned int)(0x10000 - (*(_DWORD *)(v7 + 16) - *(_DWORD *)(v7 + 24))) <= 0x3F )
  {
    v79 = a5;
    sub_772E70((_QWORD *)(v7 + 16));
    v20 = qword_4F080A8;
    v19 = *(_QWORD *)(v7 + 16);
    a5 = v79;
  }
  *(_QWORD *)(v7 + 16) = v19 + 64;
  v71 = v19 + 16;
  *(_QWORD *)v19 = 0;
  *(_QWORD *)(v19 + 16) = a4;
  *(_BYTE *)(v19 + 7) |= 1u;
  v21 = (__int64)v84;
  *(_OWORD *)(v19 + 24) = 0;
  *(_QWORD *)(v19 + 8) = v20;
  *(_QWORD *)(v19 + 40) = a5;
  *(_DWORD *)(v19 + 28) = v80;
  if ( (*(_BYTE *)(v21 + 192) & 2) != 0 && (a6 & 1) == 0 )
  {
    v83 = a5;
    v62 = sub_771080((__int64 *)&v84, v71, &v85);
    a5 = v83;
    if ( !v62 )
    {
      if ( (*(_BYTE *)(v7 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0xA8Du, a3, (_QWORD *)(v7 + 96));
        sub_770D30(v7);
      }
      return 0;
    }
    v21 = (__int64)v84;
    a4 = *(_QWORD **)(v19 + 16);
  }
  v76 = a5;
  v22 = sub_72B840(v21);
  v23 = v84;
  v24 = v76;
  v25 = v22;
  if ( (*(_BYTE *)(v22 + 29) & 8) == 0 )
  {
    if ( (*(_BYTE *)(v7 + 132) & 0x20) == 0 )
    {
      sub_686E10(0xA7Cu, a3, *v84, (_QWORD *)(v7 + 96));
      sub_770D30(v7);
    }
    return 0;
  }
  v26 = *(_QWORD *)(v22 + 80);
  v77 = *(_QWORD *)(v84[5] + 32);
  v27 = qword_4D042E0 / unk_4D042E8 + 1LL;
  *(_QWORD *)(v7 + 120) += v27;
  v28 = *(_QWORD *)(v25 + 64);
  v65 = v27;
  if ( !v28 )
    return 0;
  *(_DWORD *)(v19 + 48) = v80;
  v29 = *(_DWORD *)(v7 + 8);
  v30 = *(_QWORD *)v7;
  v64 = v28 >> 3;
  for ( i = v29 & (v28 >> 3); ; i = v29 & (i + 1) )
  {
    v32 = i;
    v33 = (unsigned __int64 *)(v30 + 16LL * i);
    if ( !*v33 )
      break;
    if ( *v33 == v28 )
    {
      v32 = v30 + 16LL * i;
      *(_QWORD *)(v19 + 56) = *(_QWORD *)(v32 + 8);
      *(_QWORD *)(v32 + 8) = v71;
      goto LABEL_25;
    }
  }
  *v33 = v28;
  v33[1] = v71;
  v34 = *(_DWORD *)(v7 + 12) + 1;
  *(_DWORD *)(v7 + 12) = v34;
  if ( 2 * v34 > v29 )
  {
    v63 = v24;
    v68 = v28;
    v70 = v25;
    v75 = v26;
    sub_7704A0(v7);
    v24 = v63;
    v28 = v68;
    v25 = v70;
    v26 = v75;
  }
  *(_QWORD *)(v19 + 56) = 0;
  v23 = v84;
LABEL_25:
  v35 = *(_QWORD *)(v7 + 72);
  v89 &= 0xF0u;
  *(_BYTE *)(v7 + 132) |= 0x80u;
  v86[0] = v35;
  v88 = 0;
  v86[3] = a4;
  LODWORD(v35) = *(_DWORD *)(v7 + 128);
  v86[1] = v23;
  v87 = v35;
  *(_QWORD *)(v7 + 72) = v86;
  v36 = *(_BYTE *)(v26 + 40) == 19;
  v86[2] = a3;
  v86[4] = v24;
  if ( v36 )
    v26 = *(_QWORD *)(*(_QWORD *)(v26 + 72) + 8LL);
  v66 = v24;
  v69 = v28;
  v72 = v25;
  v37 = sub_7987E0(v7, *(_QWORD *)(v26 + 72), v25, v25, v24, v32);
  v38 = v66;
  v39 = v69;
  v10 = v37;
  *(_BYTE *)(v66 - 9) &= ~1u;
  v40 = *(__int64 **)(v72 + 48);
  if ( v40 )
  {
    v67 = v37;
    v41 = v38;
    v73 = v19;
    v42 = v7;
    v43 = v40;
    while ( 1 )
    {
      v44 = v43[2];
      v45 = qword_4F08388 & (v44 >> 3);
      v46 = qword_4F08380 + 16LL * v45;
      v47 = *(_QWORD *)v46;
      if ( *((_BYTE *)v43 + 8) == 2 )
      {
        if ( v44 == v47 )
        {
LABEL_77:
          v48 = (_QWORD *)((char *)a4 + *(unsigned int *)(v46 + 8));
        }
        else
        {
          while ( v47 )
          {
            v45 = qword_4F08388 & (v45 + 1);
            v46 = qword_4F08380 + 16LL * v45;
            v47 = *(_QWORD *)v46;
            if ( *(_QWORD *)v46 == v44 )
              goto LABEL_77;
          }
          v48 = a4;
        }
        v49 = v43[3];
        if ( *(_BYTE *)(v77 + 140) == 11 )
          *a4 = 0;
      }
      else
      {
        if ( v44 == v47 )
        {
LABEL_75:
          v48 = (_QWORD *)((char *)a4 + *(unsigned int *)(v46 + 8));
        }
        else
        {
          while ( v47 )
          {
            v45 = qword_4F08388 & (v45 + 1);
            v46 = qword_4F08380 + 16LL * v45;
            v47 = *(_QWORD *)v46;
            if ( *(_QWORD *)v46 == v44 )
              goto LABEL_75;
          }
          v48 = a4;
        }
        v49 = v43[3];
      }
      *v48 = 0;
      if ( !(unsigned int)sub_798FD0(v42, *(_QWORD *)(v49 + 16), a3, v48, v41, 1) )
        break;
      *(_BYTE *)(v41 + -(((unsigned int)((_DWORD)v48 - v41) >> 3) + 10)) &= ~(1 << (((_BYTE)v48 - v41) & 7));
      v43 = (__int64 *)*v43;
      if ( !v43 )
      {
        v7 = v42;
        v38 = v41;
        v39 = v69;
        v19 = v73;
        v10 = v67;
        goto LABEL_39;
      }
    }
    v7 = v42;
    v38 = v41;
    v10 = 0;
    v39 = v69;
    v19 = v73;
  }
LABEL_39:
  v74 = v39;
  if ( !(unsigned int)sub_777E50(v7, (int)a4, v77, v38) )
    return 0;
  v50 = *(_DWORD *)(v7 + 8);
  v51 = *(_QWORD *)v7;
  v52 = v50 & v64;
  *(_QWORD *)(v7 + 72) = **(_QWORD **)(v7 + 72);
  v53 = *(_QWORD *)(v19 + 56);
  v54 = (_QWORD *)(v51 + 16LL * (v50 & (unsigned int)v64));
  if ( v53 )
  {
    for ( ; v74 != *v54; v54 = (_QWORD *)(v51 + 16LL * v52) )
      v52 = v50 & (v52 + 1);
    v54[1] = v53;
  }
  else
  {
    while ( *v54 != v74 )
    {
      v52 = v50 & (v52 + 1);
      v54 = (_QWORD *)(v51 + 16LL * v52);
    }
    *v54 = 0;
    if ( *(_QWORD *)(v51 + 16LL * ((v52 + 1) & v50)) )
      sub_771200(*(_QWORD *)v7, *(_DWORD *)(v7 + 8), v52);
    --*(_DWORD *)(v7 + 12);
  }
  v55 = *(_DWORD *)(v7 + 64);
  v56 = *(_QWORD *)(v7 + 56);
  v57 = v55 & v80;
  for ( j = (_DWORD *)(v56 + 4LL * (v55 & v80)); v80 != *j; j = (_DWORD *)(v56 + 4LL * v57) )
    v57 = v55 & (v57 + 1);
  *j = 0;
  if ( *(_DWORD *)(v56 + 4LL * ((v57 + 1) & v55)) )
    sub_771390(*(_QWORD *)(v7 + 56), *(_DWORD *)(v7 + 64), v57);
  --*(_DWORD *)(v7 + 68);
  *(_BYTE *)(v7 + 132) |= 0x80u;
  *(_QWORD *)(v7 + 120) += 2 - v65;
  return v10;
}
