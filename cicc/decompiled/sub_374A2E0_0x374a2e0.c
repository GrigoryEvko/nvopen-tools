// Function: sub_374A2E0
// Address: 0x374a2e0
//
__int64 __fastcall sub_374A2E0(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rdx
  __int64 v6; // rbx
  unsigned __int8 **v7; // rbx
  __int64 v8; // rdx
  unsigned __int64 v9; // r12
  signed __int64 v10; // r14
  unsigned __int64 v11; // r15
  unsigned __int8 *v12; // r11
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r11
  unsigned __int8 v16; // al
  char **v17; // rsi
  int v18; // eax
  bool v19; // al
  __int64 v20; // r9
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  unsigned __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // r10
  unsigned int v29; // eax
  __int64 v30; // r9
  unsigned __int64 v31; // rdi
  __int64 v32; // r15
  unsigned __int64 v33; // r8
  char v34; // r15
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r11
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // r15
  unsigned int v40; // ecx
  __int64 (*v41)(); // rax
  unsigned __int64 v42; // r11
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  unsigned int v54; // [rsp+8h] [rbp-88h]
  __int64 v55; // [rsp+10h] [rbp-80h]
  __int64 v56; // [rsp+18h] [rbp-78h]
  char v57; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v58; // [rsp+20h] [rbp-70h]
  __int64 v59; // [rsp+20h] [rbp-70h]
  __int64 v60; // [rsp+20h] [rbp-70h]
  unsigned int v61; // [rsp+28h] [rbp-68h]
  __int64 v62; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v63; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v64; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v65; // [rsp+28h] [rbp-68h]
  char **v66; // [rsp+30h] [rbp-60h]
  __int64 v67; // [rsp+30h] [rbp-60h]
  unsigned __int8 *v68; // [rsp+30h] [rbp-60h]
  __int64 v69; // [rsp+30h] [rbp-60h]
  __int64 v70; // [rsp+30h] [rbp-60h]
  unsigned __int64 v71; // [rsp+30h] [rbp-60h]
  unsigned __int8 *v72; // [rsp+30h] [rbp-60h]
  __int64 v73; // [rsp+30h] [rbp-60h]
  __int64 v74; // [rsp+30h] [rbp-60h]
  __int64 v75; // [rsp+30h] [rbp-60h]
  __int64 v77; // [rsp+40h] [rbp-50h]
  unsigned int v78; // [rsp+4Ch] [rbp-44h]
  unsigned __int64 v79; // [rsp+50h] [rbp-40h] BYREF
  __int64 v80; // [rsp+58h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v78 = sub_3746830(a1, *v3);
  if ( !v78 )
    return 0;
  v4 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    return 0;
  v54 = sub_2D5BAE0(a1[16], a1[14], (__int64 *)v4, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(a2 - 8);
  else
    v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v7 = (unsigned __int8 **)(v6 + 32);
  v77 = a2;
  v9 = sub_BB5290(a2) & 0xFFFFFFFFFFFFFFF9LL | 4;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v77 = *(_QWORD *)(a2 - 8) + 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( v7 == (unsigned __int8 **)v77 )
  {
LABEL_82:
    sub_3742B00((__int64)a1, (_BYTE *)a2, v78, 1);
    return 1;
  }
  v10 = v9;
  v11 = 0;
  do
  {
    v12 = *v7;
    if ( v10 )
    {
      if ( (v10 & 6) == 0 )
      {
        v13 = v10 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v8 = *((_QWORD *)v12 + 3);
          v14 = v8;
          if ( *((_DWORD *)v12 + 8) > 0x40u )
            v14 = *(_QWORD *)v8;
          v15 = v13;
          if ( v14 )
          {
            v71 = v13;
            v43 = 16LL * (unsigned int)v14 + sub_AE4AC0(a1[14], v13) + 24;
            v44 = *(_QWORD *)v43;
            LOBYTE(v43) = *(_BYTE *)(v43 + 8);
            v79 = v44;
            LOBYTE(v80) = v43;
            v45 = sub_CA1930(&v79);
            v15 = v71;
            v11 += v45;
            if ( v11 > 0x7FF )
            {
              v78 = sub_3749CE0((__int64 **)a1, (unsigned __int16)v54, 0x38u, v78, v11, (unsigned __int16)v54);
              if ( !v78 )
                return 0;
              v15 = v71;
              v11 = 0;
            }
          }
LABEL_21:
          v15 = sub_BCBAE0(v15, *v7, v8);
          goto LABEL_22;
        }
      }
    }
    if ( *v12 != 17 )
    {
      if ( v11 )
      {
        v68 = *v7;
        v29 = sub_3749CE0((__int64 **)a1, (unsigned __int16)v54, 0x38u, v78, v11, (unsigned __int16)v54);
        v12 = v68;
        v78 = v29;
        if ( !v29 )
          return 0;
      }
      v30 = a1[14];
      v31 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      v32 = (v10 >> 1) & 3;
      v33 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v10 )
      {
        if ( v32 == 2 )
        {
          if ( v31 )
          {
LABEL_52:
            v58 = v12;
            v62 = v33;
            v69 = v30;
            v34 = sub_AE5020(v30, v33);
            v35 = sub_9208B0(v69, v62);
            v37 = (__int64)v58;
            v80 = v36;
            v38 = ((1LL << v34) + ((unsigned __int64)(v35 + 7) >> 3) - 1) >> v34 << v34;
            goto LABEL_53;
          }
LABEL_71:
          v63 = v12;
          v73 = a1[14];
          v47 = sub_BCBAE0(v31, *v7, v8);
          v30 = v73;
          v12 = v63;
          v33 = v47;
          goto LABEL_52;
        }
        if ( v32 != 1 )
          goto LABEL_71;
        if ( v31 )
        {
          v33 = *(_QWORD *)(v31 + 24);
        }
        else
        {
          v65 = v12;
          v75 = a1[14];
          v52 = sub_BCBAE0(0, *v7, v8);
          v30 = v75;
          v12 = v65;
          v33 = v52;
        }
      }
      else
      {
        v64 = v12;
        v74 = a1[14];
        v50 = sub_BCBAE0(v31, *v7, v8);
        v30 = v74;
        v12 = v64;
        v33 = v50;
        if ( v32 != 1 )
          goto LABEL_52;
      }
      v72 = v12;
      v46 = sub_9208B0(v30, v33);
      v37 = (__int64)v72;
      v80 = v36;
      v38 = (unsigned __int64)(v46 + 7) >> 3;
LABEL_53:
      v70 = v37;
      LOBYTE(v80) = v36;
      v79 = v38;
      v39 = sub_CA1930(&v79);
      v40 = sub_3746A00(a1, (unsigned __int16)v54, v70);
      if ( !v40 )
        return 0;
      if ( v39 != 1 )
      {
        v40 = sub_3749CE0((__int64 **)a1, (unsigned __int16)v54, 0x3Au, v40, v39, (unsigned __int16)v54);
        if ( !v40 )
          return 0;
      }
      v41 = *(__int64 (**)())(*a1 + 72);
      if ( v41 == sub_3740EF0 )
        return 0;
      v78 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, _QWORD, _QWORD))v41)(
              a1,
              (unsigned __int16)v54,
              (unsigned __int16)v54,
              56,
              v78,
              v40);
      if ( !v78 )
        return 0;
LABEL_39:
      v11 = 0;
      goto LABEL_40;
    }
    v8 = *((unsigned int *)v12 + 8);
    v17 = (char **)(v12 + 24);
    if ( (unsigned int)v8 <= 0x40 )
    {
      v19 = *((_QWORD *)v12 + 3) == 0;
    }
    else
    {
      v61 = *((_DWORD *)v12 + 8);
      v66 = (char **)(v12 + 24);
      v18 = sub_C444A0((__int64)(v12 + 24));
      v8 = v61;
      v17 = v66;
      v19 = v61 == v18;
    }
    if ( !v19 )
    {
      sub_C44B10((__int64)&v79, v17, 0x40u);
      if ( (unsigned int)v80 > 0x40 )
      {
        v67 = *(_QWORD *)v79;
        j_j___libc_free_0_0(v79);
      }
      else
      {
        v67 = 0;
        if ( (_DWORD)v80 )
          v67 = (__int64)(v79 << (64 - (unsigned __int8)v80)) >> (64 - (unsigned __int8)v80);
      }
      v20 = a1[14];
      v21 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      v22 = (v10 >> 1) & 3;
      v23 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v10 )
      {
        if ( v22 == 2 )
        {
          if ( v21 )
          {
LABEL_36:
            v55 = v23;
            v56 = v20;
            v57 = sub_AE5020(v20, v23);
            v24 = sub_9208B0(v56, v55);
            v80 = v25;
            v26 = ((1LL << v57) + ((unsigned __int64)(v24 + 7) >> 3) - 1) >> v57 << v57;
LABEL_37:
            LOBYTE(v80) = v25;
            v79 = v67 * v26;
            v11 += sub_CA1930(&v79);
            if ( v11 <= 0x7FF )
              goto LABEL_40;
            v78 = sub_3749CE0((__int64 **)a1, (unsigned __int16)v54, 0x38u, v78, v11, (unsigned __int16)v54);
            if ( !v78 )
              return 0;
            goto LABEL_39;
          }
        }
        else if ( v22 == 1 )
        {
          if ( v21 )
          {
            v48 = *(_QWORD *)(v21 + 24);
          }
          else
          {
            v60 = a1[14];
            v53 = sub_BCBAE0(0, *v7, 1);
            v20 = v60;
            v48 = v53;
          }
          v49 = sub_9208B0(v20, v48);
          v80 = v25;
          v26 = (unsigned __int64)(v49 + 7) >> 3;
          goto LABEL_37;
        }
      }
      v59 = a1[14];
      v51 = sub_BCBAE0(v21, *v7, v22);
      v20 = v59;
      v23 = v51;
      goto LABEL_36;
    }
LABEL_40:
    v27 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    v15 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v10 )
      goto LABEL_21;
    v28 = (v10 >> 1) & 3;
    if ( v28 == 2 )
    {
      if ( !v27 )
        goto LABEL_21;
    }
    else
    {
      if ( v28 != 1 || !v27 )
        goto LABEL_21;
      v15 = *(_QWORD *)(v27 + 24);
    }
LABEL_22:
    v16 = *(_BYTE *)(v15 + 8);
    if ( v16 == 16 )
    {
      v10 = *(_QWORD *)(v15 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else
    {
      v8 = (unsigned int)v16 - 17;
      if ( (unsigned int)v8 > 1 )
      {
        v42 = v15 & 0xFFFFFFFFFFFFFFF9LL;
        v10 = 0;
        if ( v16 == 15 )
          v10 = v42;
      }
      else
      {
        v10 = v15 & 0xFFFFFFFFFFFFFFF9LL | 2;
      }
    }
    v7 += 4;
  }
  while ( (unsigned __int8 **)v77 != v7 );
  if ( !v11 )
    goto LABEL_82;
  v78 = sub_3749CE0((__int64 **)a1, v54, 0x38u, v78, v11, v54);
  if ( v78 )
    goto LABEL_82;
  return 0;
}
