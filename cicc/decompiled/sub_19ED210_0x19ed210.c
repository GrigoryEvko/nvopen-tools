// Function: sub_19ED210
// Address: 0x19ed210
//
__int64 __fastcall sub_19ED210(__int64 a1, __int64 ****a2, __int64 a3, __int64 ***a4, __int64 a5)
{
  unsigned __int64 v6; // r13
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned int v11; // eax
  char v12; // cl
  _QWORD *v13; // rax
  _QWORD **v14; // rdx
  __int64 ****v15; // rbx
  __int64 **v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 ****v19; // rax
  __int64 ****v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rcx
  __int64 *v27; // rdi
  __int64 v28; // rbx
  __int64 *v29; // rax
  unsigned __int8 v30; // si
  __int64 *v31; // r10
  int v33; // eax
  int v34; // esi
  __int64 v35; // rcx
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rsi
  char v41; // al
  __int64 v42; // rsi
  char v43; // al
  __int64 v44; // rax
  __int64 *v45; // rdx
  __int64 v46; // rcx
  __int64 *v47; // r13
  __int64 *v48; // r15
  __int64 v49; // rax
  unsigned int v50; // r15d
  int j; // eax
  int v52; // r10d
  __int64 v54; // [rsp+20h] [rbp-E0h]
  __int64 v55; // [rsp+20h] [rbp-E0h]
  __int64 v56; // [rsp+28h] [rbp-D8h]
  __int64 ****v57; // [rsp+28h] [rbp-D8h]
  __int64 *v58; // [rsp+28h] [rbp-D8h]
  char v59; // [rsp+3Eh] [rbp-C2h] BYREF
  char v60; // [rsp+3Fh] [rbp-C1h] BYREF
  __int64 ***v61; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+48h] [rbp-B8h] BYREF
  __int64 ****v63; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v64; // [rsp+58h] [rbp-A8h]
  __int64 ****v65; // [rsp+60h] [rbp-A0h]
  __int64 v66; // [rsp+68h] [rbp-98h]
  __int64 *v67; // [rsp+70h] [rbp-90h]
  char *v68; // [rsp+78h] [rbp-88h]
  char *v69; // [rsp+80h] [rbp-80h]
  __int64 ****v70; // [rsp+90h] [rbp-70h] BYREF
  __int64 ****v71; // [rsp+98h] [rbp-68h]
  __int64 ****v72; // [rsp+A0h] [rbp-60h]
  __int64 v73; // [rsp+A8h] [rbp-58h]
  __int64 *v74; // [rsp+B0h] [rbp-50h]
  char *v75; // [rsp+B8h] [rbp-48h]
  char *i; // [rsp+C0h] [rbp-40h]

  v6 = (unsigned int)a3;
  v62 = a5;
  v59 = 0;
  v60 = 1;
  v61 = a4;
  v8 = sub_145CDC0(0x38u, (__int64 *)(a1 + 64));
  v9 = v8;
  if ( v8 )
  {
    *(_DWORD *)(v8 + 32) = a3;
    *(_QWORD *)(v8 + 8) = 0xFFFFFFFD00000008LL;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)v8 = &unk_49F4ED8;
    v10 = v62;
    *(_DWORD *)(v9 + 36) = 0;
    *(_QWORD *)(v9 + 40) = 0;
    *(_QWORD *)(v9 + 48) = v10;
  }
  else
  {
    v6 = MEMORY[0x20];
  }
  if ( !v6 )
  {
    v12 = 0;
    goto LABEL_8;
  }
  if ( !--v6 )
  {
    v12 = 0;
LABEL_8:
    if ( !*(_DWORD *)(a1 + 176) )
      goto LABEL_6;
    goto LABEL_9;
  }
  _BitScanReverse64(&v6, v6);
  v11 = 64 - (v6 ^ 0x3F);
  v12 = 64 - (v6 ^ 0x3F);
  v6 = v11;
  if ( *(_DWORD *)(a1 + 176) <= v11 )
  {
LABEL_6:
    v13 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 64), 8LL << v12, 8);
    goto LABEL_11;
  }
LABEL_9:
  v14 = (_QWORD **)(*(_QWORD *)(a1 + 168) + 8 * v6);
  v13 = *v14;
  if ( !*v14 )
    goto LABEL_6;
  *v14 = (_QWORD *)*v13;
LABEL_11:
  *(_QWORD *)(v9 + 24) = v13;
  v15 = &a2[2 * a3];
  v16 = **a2;
  v70 = v15;
  *(_DWORD *)(v9 + 12) = 53;
  *(_QWORD *)(v9 + 40) = v16;
  v71 = v15;
  v72 = &v61;
  v74 = &v62;
  v75 = &v60;
  i = &v59;
  v73 = a1;
  sub_19E91F0((__int64)&v70);
  v63 = a2;
  v64 = (__int64 *)v15;
  v68 = &v60;
  v69 = &v59;
  v65 = &v61;
  v66 = a1;
  v67 = &v62;
  sub_19E91F0((__int64)&v63);
  v19 = v63;
  v20 = v70;
  v71 = (__int64 ****)v64;
  v70 = v63;
  v72 = v65;
  v73 = v66;
  v74 = v67;
  v75 = v68;
  for ( i = v69; v70 != v20; v19 = v70 )
  {
    v21 = sub_19E1ED0(a1, *v19);
    v22 = *(_QWORD *)(v9 + 24);
    v23 = v21;
    v24 = *(unsigned int *)(v9 + 36);
    *(_DWORD *)(v9 + 36) = v24 + 1;
    *(_QWORD *)(v22 + 8 * v24) = v23;
    v70 += 2;
    sub_19E91F0((__int64)&v70);
  }
  v25 = *(__int64 **)(v9 + 24);
  v26 = 0;
  v27 = &v25[*(unsigned int *)(v9 + 36)];
  if ( v27 == v25 )
  {
    sub_19E1860(v27, *(_DWORD *)(v9 + 32), a1 + 168, 0, v17, v18);
    return *(_QWORD *)(a1 + 2088);
  }
  while ( 1 )
  {
    v28 = *v25;
    v29 = v25 + 1;
    v30 = *(_BYTE *)(*v25 + 16);
    if ( v30 != 9 )
      break;
    v26 = 1;
    if ( v27 == v29 )
    {
      v49 = sub_1599EF0(*a4);
      return sub_19E59B0(a1, v49);
    }
    ++v25;
  }
  v31 = v25 + 1;
  if ( v27 == v29 )
  {
LABEL_17:
    if ( v27 == v25 )
      goto LABEL_18;
  }
  else
  {
    do
    {
      if ( *(_BYTE *)(*v31 + 16) != 9 )
        goto LABEL_17;
      ++v31;
      v26 = 1;
    }
    while ( v27 != v31 );
    if ( v27 == v25 )
      goto LABEL_28;
  }
  while ( v27 != v29 )
  {
    while ( *(_BYTE *)(*v29 + 16) == 9 )
    {
      ++v29;
      v26 = 1;
      if ( v27 == v29 )
        goto LABEL_28;
    }
    if ( v27 == v29 )
      break;
    if ( v28 != *v29 )
      return v9;
    ++v29;
  }
LABEL_18:
  if ( !(_BYTE)v26 )
    goto LABEL_19;
LABEL_28:
  if ( v59 && !v60 )
  {
    if ( !(unsigned __int8)sub_19ECD70(a1, (__int64)a4) )
      return v9;
    v30 = *(_BYTE *)(v28 + 16);
  }
  if ( v30 <= 0x17u )
    goto LABEL_20;
  v33 = *(_DWORD *)(a1 + 1496);
  if ( v33 )
  {
    v34 = v33 - 1;
    v35 = *(_QWORD *)(a1 + 1480);
    v36 = (v33 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v37 = (__int64 *)(v35 + 16LL * v36);
    v38 = *v37;
    if ( *v37 != v28 )
    {
      for ( j = 1; ; j = v52 )
      {
        if ( v38 == -8 )
          return v9;
        v52 = j + 1;
        v36 = v34 & (j + v36);
        v37 = (__int64 *)(v35 + 16LL * v36);
        v38 = *v37;
        if ( v28 == *v37 )
          break;
      }
    }
    v39 = v37[1];
    if ( v39 )
    {
      v40 = *(_QWORD *)(v39 + 8);
      if ( *(_BYTE *)(v40 + 16) <= 0x11u )
        goto LABEL_53;
      v56 = v37[1];
      v41 = sub_15CCEE0(*(_QWORD *)(a1 + 8), v40, (__int64)a4);
      v26 = v56;
      if ( !v41 )
      {
        v42 = *(_QWORD *)(v56 + 16);
        if ( !v42 || (v43 = sub_15CCEE0(*(_QWORD *)(a1 + 8), v42, (__int64)a4), v26 = v56, !v43) )
        {
          v54 = v26;
          v57 = (__int64 ****)(v26 + 56);
          sub_19E54A0(&v70, (__int64 *)(v26 + 56));
          v44 = *(_QWORD *)(v54 + 72);
          if ( v44 == *(_QWORD *)(v54 + 64) )
            v45 = (__int64 *)(v44 + 8LL * *(unsigned int *)(v54 + 84));
          else
            v45 = (__int64 *)(v44 + 8LL * *(unsigned int *)(v54 + 80));
          v63 = *(__int64 *****)(v54 + 72);
          v64 = v45;
          sub_19E4730((__int64)&v63);
          v46 = v54;
          v47 = (__int64 *)v63;
          v48 = v64;
          v65 = v57;
          v66 = *(_QWORD *)(v54 + 56);
          v58 = (__int64 *)v70;
          if ( v63 == v70 )
            return v9;
          while ( 1 )
          {
            if ( *v47 != *(_QWORD *)(v46 + 8) )
            {
              v55 = v46;
              if ( sub_15CCEE0(*(_QWORD *)(a1 + 8), *v47, (__int64)a4) )
              {
                if ( v58 != v47 )
                  break;
                return v9;
              }
              v46 = v55;
            }
            do
              ++v47;
            while ( v48 != v47 && (unsigned __int64)*v47 >= 0xFFFFFFFFFFFFFFFELL );
            if ( v58 == v47 )
              return v9;
          }
        }
      }
      v30 = *(_BYTE *)(v28 + 16);
LABEL_19:
      if ( v30 <= 0x17u )
      {
LABEL_20:
        sub_19E1860(*(_QWORD **)(v9 + 24), *(_DWORD *)(v9 + 32), a1 + 168, v26, v17, v18);
        return sub_19E5BE0(a1, v28);
      }
LABEL_53:
      v50 = sub_19E5210(a1 + 2392, v28);
      if ( (unsigned int)sub_19E5210(a1 + 2392, (__int64)a4) < v50 )
        return v9;
      goto LABEL_20;
    }
  }
  return v9;
}
