// Function: sub_10B5C50
// Address: 0x10b5c50
//
unsigned __int8 *__fastcall sub_10B5C50(_BYTE *a1, __int64 a2)
{
  __int64 *v2; // rdx
  bool v4; // zf
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // eax
  int v21; // r14d
  __int64 v22; // rbx
  __int64 v23; // r15
  unsigned int v24; // eax
  int v25; // r14d
  __int64 v26; // rbx
  __int64 v27; // r11
  unsigned __int8 *v28; // r14
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned int *v39; // rbx
  __int64 v40; // r14
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r11
  unsigned int *v46; // rbx
  __int64 v47; // r14
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 v50; // rdx
  __int64 v51; // [rsp+8h] [rbp-C8h]
  __int64 v52; // [rsp+8h] [rbp-C8h]
  __int64 v53; // [rsp+8h] [rbp-C8h]
  __int64 v54; // [rsp+8h] [rbp-C8h]
  __int64 v55; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v56; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v57; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v58; // [rsp+30h] [rbp-A0h]
  __int64 v59; // [rsp+38h] [rbp-98h]
  _BYTE v60[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v61; // [rsp+60h] [rbp-70h]
  __int64 *v62; // [rsp+70h] [rbp-60h] BYREF
  __int64 v63; // [rsp+78h] [rbp-58h] BYREF
  __int64 *v64; // [rsp+80h] [rbp-50h]
  __int64 *v65; // [rsp+88h] [rbp-48h]
  __int64 *v66; // [rsp+90h] [rbp-40h]

  v2 = &v56;
  v4 = *a1 == 43;
  v62 = &v56;
  v65 = &v55;
  v63 = 0x3FF0000000000000LL;
  v64 = &v57;
  v66 = &v57;
  if ( !v4 )
    return sub_10A0280(a1, (unsigned int **)a2);
  v6 = *((_QWORD *)a1 - 8);
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 || *(_QWORD *)(v7 + 8) || *(_BYTE *)v6 != 47 )
    goto LABEL_4;
  if ( !*(_QWORD *)(v6 - 64) )
  {
LABEL_68:
    v29 = *(_QWORD *)(v6 - 32);
    if ( !v29 )
      goto LABEL_4;
    v2 = v62;
    goto LABEL_31;
  }
  v56 = *(_QWORD *)(v6 - 64);
  v29 = *(_QWORD *)(v6 - 32);
  v30 = *(_QWORD *)(v29 + 16);
  if ( v30 && !*(_QWORD *)(v30 + 8) && *(_BYTE *)v29 == 45 )
  {
    if ( (unsigned __int8)sub_1009690((double *)&v63, *(_QWORD *)(v29 - 64)) )
    {
      v50 = *(_QWORD *)(v29 - 32);
      if ( v50 )
      {
        *v64 = v50;
        goto LABEL_37;
      }
    }
    goto LABEL_68;
  }
LABEL_31:
  *v2 = v29;
  v31 = *(_QWORD *)(v6 - 64);
  v32 = *(_QWORD *)(v31 + 16);
  if ( !v32 )
    goto LABEL_4;
  if ( *(_QWORD *)(v32 + 8) )
    goto LABEL_4;
  if ( *(_BYTE *)v31 != 45 )
    goto LABEL_4;
  if ( !(unsigned __int8)sub_1009690((double *)&v63, *(_QWORD *)(v31 - 64)) )
    goto LABEL_4;
  v33 = *(_QWORD *)(v31 - 32);
  if ( !v33 )
    goto LABEL_4;
  *v64 = v33;
LABEL_37:
  v8 = *((_QWORD *)a1 - 4);
  v34 = *(_QWORD *)(v8 + 16);
  if ( !v34 || *(_QWORD *)(v34 + 8) )
    return sub_10A0280(a1, (unsigned int **)a2);
  if ( *(_BYTE *)v8 != 47 )
    goto LABEL_6;
  v35 = *(_QWORD *)(v8 - 64);
  if ( v35 )
  {
    *v65 = v35;
    v36 = *(_QWORD *)(v8 - 32);
    if ( v36 == *v66 )
      goto LABEL_21;
    if ( !v36 )
      goto LABEL_4;
  }
  else
  {
    v36 = *(_QWORD *)(v8 - 32);
    if ( !v36 )
      goto LABEL_6;
  }
  *v65 = v36;
  if ( *(_QWORD *)(v8 - 64) == *v66 )
    goto LABEL_21;
LABEL_4:
  v8 = *((_QWORD *)a1 - 4);
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 || *(_QWORD *)(v9 + 8) )
    return sub_10A0280(a1, (unsigned int **)a2);
LABEL_6:
  if ( *(_BYTE *)v8 != 47 )
    return sub_10A0280(a1, (unsigned int **)a2);
  v10 = *(_QWORD *)(v8 - 64);
  if ( !v10 )
  {
LABEL_66:
    v11 = *(_QWORD *)(v8 - 32);
    if ( !v11 )
      return sub_10A0280(a1, (unsigned int **)a2);
    goto LABEL_9;
  }
  *v62 = v10;
  v11 = *(_QWORD *)(v8 - 32);
  v12 = *(_QWORD *)(v11 + 16);
  if ( v12 && !*(_QWORD *)(v12 + 8) && *(_BYTE *)v11 == 45 )
  {
    if ( (unsigned __int8)sub_1009690((double *)&v63, *(_QWORD *)(v11 - 64)) )
    {
      v15 = *(_QWORD *)(v11 - 32);
      if ( v15 )
        goto LABEL_14;
    }
    goto LABEL_66;
  }
LABEL_9:
  *v62 = v11;
  v13 = *(_QWORD *)(v8 - 64);
  v14 = *(_QWORD *)(v13 + 16);
  if ( !v14 )
    return sub_10A0280(a1, (unsigned int **)a2);
  if ( *(_QWORD *)(v14 + 8) )
    return sub_10A0280(a1, (unsigned int **)a2);
  if ( *(_BYTE *)v13 != 45 )
    return sub_10A0280(a1, (unsigned int **)a2);
  if ( !(unsigned __int8)sub_1009690((double *)&v63, *(_QWORD *)(v13 - 64)) )
    return sub_10A0280(a1, (unsigned int **)a2);
  v15 = *(_QWORD *)(v13 - 32);
  if ( !v15 )
    return sub_10A0280(a1, (unsigned int **)a2);
LABEL_14:
  *v64 = v15;
  v16 = *((_QWORD *)a1 - 8);
  v17 = *(_QWORD *)(v16 + 16);
  if ( !v17 || *(_QWORD *)(v17 + 8) || *(_BYTE *)v16 != 47 )
    return sub_10A0280(a1, (unsigned int **)a2);
  v18 = *(_QWORD *)(v16 - 64);
  if ( v18 )
  {
    *v65 = v18;
    v19 = *(_QWORD *)(v16 - 32);
    if ( v19 == *v66 )
      goto LABEL_21;
  }
  else
  {
    v19 = *(_QWORD *)(v16 - 32);
  }
  if ( !v19 )
    return sub_10A0280(a1, (unsigned int **)a2);
  *v65 = v19;
  if ( *(_QWORD *)(v16 - 64) != *v66 )
    return sub_10A0280(a1, (unsigned int **)a2);
LABEL_21:
  v61 = 257;
  v20 = sub_B45210((__int64)a1);
  BYTE4(v58) = 1;
  v4 = *(_BYTE *)(a2 + 108) == 0;
  LODWORD(v58) = v20;
  v21 = v20;
  v22 = v55;
  v59 = v58;
  if ( v4 )
  {
    v51 = v56;
    v23 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a2 + 80) + 40LL))(
            *(_QWORD *)(a2 + 80),
            16,
            v55,
            v56,
            v20);
    if ( !v23 )
    {
      LOWORD(v66) = 257;
      v37 = sub_B504D0(16, v22, v51, (__int64)&v62, 0, 0);
      v38 = *(_QWORD *)(a2 + 96);
      v23 = v37;
      if ( v38 )
        sub_B99FD0(v37, 3u, v38);
      sub_B45150(v23, v21);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v23,
        v60,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v39 = *(unsigned int **)a2;
      v40 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v40 )
      {
        do
        {
          v41 = *((_QWORD *)v39 + 1);
          v42 = *v39;
          v39 += 4;
          sub_B99FD0(v23, v42, v41);
        }
        while ( (unsigned int *)v40 != v39 );
      }
    }
  }
  else
  {
    v23 = sub_B35400(a2, 0x73u, v55, v56, v58, (__int64)v60, 0, 0, 0);
  }
  v61 = 257;
  v24 = sub_B45210((__int64)a1);
  BYTE4(v58) = 1;
  v4 = *(_BYTE *)(a2 + 108) == 0;
  LODWORD(v58) = v24;
  v25 = v24;
  v26 = v57;
  v59 = v58;
  if ( v4 )
  {
    v27 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a2 + 80) + 40LL))(
            *(_QWORD *)(a2 + 80),
            18,
            v57,
            v23,
            v24);
    if ( !v27 )
    {
      LOWORD(v66) = 257;
      v43 = sub_B504D0(18, v26, v23, (__int64)&v62, 0, 0);
      v44 = *(_QWORD *)(a2 + 96);
      v45 = v43;
      if ( v44 )
      {
        v52 = v43;
        sub_B99FD0(v43, 3u, v44);
        v45 = v52;
      }
      v53 = v45;
      sub_B45150(v45, v25);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v53,
        v60,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v46 = *(unsigned int **)a2;
      v27 = v53;
      v47 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v47 )
      {
        do
        {
          v48 = *((_QWORD *)v46 + 1);
          v49 = *v46;
          v46 += 4;
          v54 = v27;
          sub_B99FD0(v27, v49, v48);
          v27 = v54;
        }
        while ( (unsigned int *)v47 != v46 );
      }
    }
  }
  else
  {
    v27 = sub_B35400(a2, 0x6Cu, v57, v23, v58, (__int64)v60, 0, 0, 0);
  }
  LOWORD(v66) = 257;
  v28 = (unsigned __int8 *)sub_B504D0(14, v56, v27, (__int64)&v62, 0, 0);
  sub_B45260(v28, (__int64)a1, 1);
  if ( !v28 )
    return sub_10A0280(a1, (unsigned int **)a2);
  return v28;
}
