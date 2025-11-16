// Function: sub_3916190
// Address: 0x3916190
//
void __fastcall sub_3916190(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v3; // rbx
  __int64 *v4; // rax
  int v5; // edx
  int v6; // r12d
  __int64 v8; // r13
  int v9; // eax
  unsigned int v10; // esi
  __int64 v11; // r10
  unsigned int v12; // r9d
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 *v17; // rbx
  int v18; // r14d
  __int64 v19; // r12
  __int64 v20; // r10
  unsigned int v21; // r9d
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r13
  unsigned int v25; // esi
  int v26; // esi
  int v27; // esi
  __int64 v28; // r9
  unsigned int v29; // edx
  int v30; // eax
  __int64 *v31; // rdi
  __int64 v32; // r8
  int v33; // ecx
  __int64 *v34; // r10
  int v35; // ecx
  __int64 *v36; // rdi
  int v37; // eax
  int v38; // eax
  int v39; // ecx
  int v40; // eax
  int v41; // esi
  int v42; // esi
  __int64 v43; // r9
  unsigned int v44; // edx
  __int64 v45; // r8
  int v46; // ecx
  __int64 *v47; // r10
  int v48; // esi
  int v49; // esi
  __int64 v50; // r9
  int v51; // ecx
  unsigned int v52; // edx
  __int64 v53; // r8
  int v54; // esi
  int v55; // esi
  __int64 v56; // r9
  int v57; // ecx
  unsigned int v58; // edx
  __int64 v59; // r8
  _BYTE *v60; // rax
  __int64 *v61; // rax
  __int64 v62; // rdx
  _QWORD *v63; // rax
  unsigned int v64; // [rsp+8h] [rbp-98h]
  unsigned int v65; // [rsp+8h] [rbp-98h]
  __int64 v66; // [rsp+10h] [rbp-90h]
  __int64 v67; // [rsp+10h] [rbp-90h]
  __int64 *v69; // [rsp+18h] [rbp-88h]
  _QWORD v70[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v71[2]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v72; // [rsp+40h] [rbp-60h]
  _QWORD v73[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v74; // [rsp+60h] [rbp-40h]

  v2 = *(__int64 **)(a2 + 88);
  v3 = *(__int64 **)(a2 + 80);
  if ( v3 == v2 )
    return;
  v4 = *(__int64 **)(a2 + 80);
  do
  {
    v5 = *(unsigned __int8 *)(v4[1] + 184);
    if ( (unsigned int)(v5 - 6) > 2 && v5 != 20 )
    {
      v60 = (_BYTE *)*v4;
      if ( (*v60 & 4) != 0 )
      {
        v61 = (__int64 *)*((_QWORD *)v60 - 1);
        v62 = *v61;
        v63 = v61 + 2;
      }
      else
      {
        v62 = 0;
        v63 = 0;
      }
      v70[0] = v63;
      v72 = 1283;
      v71[0] = "indirect symbol '";
      v71[1] = v70;
      v70[1] = v62;
      v73[0] = v71;
      v73[1] = "' not in a symbol pointer or stub section";
      v74 = 770;
      sub_16BCFB0((__int64)v73, 1u);
    }
    v4 += 2;
  }
  while ( v2 != v4 );
  v66 = a1 + 48;
  v6 = 0;
  do
  {
    while ( 1 )
    {
      v8 = v3[1];
      v9 = *(unsigned __int8 *)(v8 + 184);
      if ( v9 == 20 || v9 == 6 )
        break;
      v3 += 2;
      ++v6;
      if ( v2 == v3 )
        goto LABEL_13;
    }
    v10 = *(_DWORD *)(a1 + 72);
    if ( v10 )
    {
      v11 = *(_QWORD *)(a1 + 56);
      v12 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_12;
      v35 = 1;
      v36 = 0;
      while ( v14 != -8 )
      {
        if ( v14 != -16 || v36 )
          v13 = v36;
        v12 = (v10 - 1) & (v35 + v12);
        v14 = *(_QWORD *)(v11 + 16LL * v12);
        if ( v8 == v14 )
          goto LABEL_12;
        ++v35;
        v36 = v13;
        v13 = (__int64 *)(v11 + 16LL * v12);
      }
      if ( !v36 )
        v36 = v13;
      v37 = *(_DWORD *)(a1 + 64);
      ++*(_QWORD *)(a1 + 48);
      v38 = v37 + 1;
      if ( 4 * v38 < 3 * v10 )
      {
        if ( v10 - *(_DWORD *)(a1 + 68) - v38 > v10 >> 3 )
          goto LABEL_36;
        v65 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
        sub_3915FD0(v66, v10);
        v54 = *(_DWORD *)(a1 + 72);
        if ( !v54 )
        {
LABEL_96:
          ++*(_DWORD *)(a1 + 64);
          BUG();
        }
        v55 = v54 - 1;
        v56 = *(_QWORD *)(a1 + 56);
        v47 = 0;
        v57 = 1;
        v58 = v55 & v65;
        v38 = *(_DWORD *)(a1 + 64) + 1;
        v36 = (__int64 *)(v56 + 16LL * (v55 & v65));
        v59 = *v36;
        if ( v8 == *v36 )
          goto LABEL_36;
        while ( v59 != -8 )
        {
          if ( v59 == -16 && !v47 )
            v47 = v36;
          v58 = v55 & (v57 + v58);
          v36 = (__int64 *)(v56 + 16LL * v58);
          v59 = *v36;
          if ( v8 == *v36 )
            goto LABEL_36;
          ++v57;
        }
        goto LABEL_53;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 48);
    }
    sub_3915FD0(v66, 2 * v10);
    v41 = *(_DWORD *)(a1 + 72);
    if ( !v41 )
      goto LABEL_96;
    v42 = v41 - 1;
    v43 = *(_QWORD *)(a1 + 56);
    v44 = v42 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v38 = *(_DWORD *)(a1 + 64) + 1;
    v36 = (__int64 *)(v43 + 16LL * v44);
    v45 = *v36;
    if ( v8 == *v36 )
      goto LABEL_36;
    v46 = 1;
    v47 = 0;
    while ( v45 != -8 )
    {
      if ( !v47 && v45 == -16 )
        v47 = v36;
      v44 = v42 & (v46 + v44);
      v36 = (__int64 *)(v43 + 16LL * v44);
      v45 = *v36;
      if ( v8 == *v36 )
        goto LABEL_36;
      ++v46;
    }
LABEL_53:
    if ( v47 )
      v36 = v47;
LABEL_36:
    *(_DWORD *)(a1 + 64) = v38;
    if ( *v36 != -8 )
      --*(_DWORD *)(a1 + 68);
    *v36 = v8;
    *((_DWORD *)v36 + 2) = v6;
LABEL_12:
    v15 = *v3;
    v3 += 2;
    ++v6;
    sub_390D5F0(a2, v15, 0);
  }
  while ( v2 != v3 );
LABEL_13:
  v16 = a2;
  v17 = *(__int64 **)(a2 + 80);
  v69 = *(__int64 **)(a2 + 88);
  if ( v69 != v17 )
  {
    v18 = 0;
    v19 = v16;
    v67 = a1 + 48;
    while ( 1 )
    {
      v24 = v17[1];
      if ( (unsigned int)*(unsigned __int8 *)(v24 + 184) - 7 <= 1 )
        break;
LABEL_18:
      ++v18;
      v17 += 2;
      if ( v69 == v17 )
        return;
    }
    v25 = *(_DWORD *)(a1 + 72);
    if ( !v25 )
    {
      ++*(_QWORD *)(a1 + 48);
      goto LABEL_22;
    }
    v20 = *(_QWORD *)(a1 + 56);
    v21 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v22 = (__int64 *)(v20 + 16LL * v21);
    v23 = *v22;
    if ( v24 == *v22 )
    {
LABEL_16:
      sub_390D5F0(v19, *v17, (bool *)v73);
      if ( LOBYTE(v73[0]) )
        *(_WORD *)(*v17 + 12) |= 1u;
      goto LABEL_18;
    }
    v39 = 1;
    v31 = 0;
    while ( v23 != -8 )
    {
      if ( v23 != -16 || v31 )
        v22 = v31;
      v21 = (v25 - 1) & (v39 + v21);
      v23 = *(_QWORD *)(v20 + 16LL * v21);
      if ( v24 == v23 )
        goto LABEL_16;
      ++v39;
      v31 = v22;
      v22 = (__int64 *)(v20 + 16LL * v21);
    }
    if ( !v31 )
      v31 = v22;
    v40 = *(_DWORD *)(a1 + 64);
    ++*(_QWORD *)(a1 + 48);
    v30 = v40 + 1;
    if ( 4 * v30 >= 3 * v25 )
    {
LABEL_22:
      sub_3915FD0(v67, 2 * v25);
      v26 = *(_DWORD *)(a1 + 72);
      if ( !v26 )
        goto LABEL_96;
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 56);
      v29 = v27 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v30 = *(_DWORD *)(a1 + 64) + 1;
      v31 = (__int64 *)(v28 + 16LL * v29);
      v32 = *v31;
      if ( v24 != *v31 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -8 )
        {
          if ( !v34 && v32 == -16 )
            v34 = v31;
          v29 = v27 & (v33 + v29);
          v31 = (__int64 *)(v28 + 16LL * v29);
          v32 = *v31;
          if ( v24 == *v31 )
            goto LABEL_45;
          ++v33;
        }
        goto LABEL_26;
      }
    }
    else if ( v25 - *(_DWORD *)(a1 + 68) - v30 <= v25 >> 3 )
    {
      v64 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
      sub_3915FD0(v67, v25);
      v48 = *(_DWORD *)(a1 + 72);
      if ( !v48 )
        goto LABEL_96;
      v49 = v48 - 1;
      v50 = *(_QWORD *)(a1 + 56);
      v34 = 0;
      v51 = 1;
      v52 = v49 & v64;
      v30 = *(_DWORD *)(a1 + 64) + 1;
      v31 = (__int64 *)(v50 + 16LL * (v49 & v64));
      v53 = *v31;
      if ( v24 != *v31 )
      {
        while ( v53 != -8 )
        {
          if ( v53 == -16 && !v34 )
            v34 = v31;
          v52 = v49 & (v51 + v52);
          v31 = (__int64 *)(v50 + 16LL * v52);
          v53 = *v31;
          if ( v24 == *v31 )
            goto LABEL_45;
          ++v51;
        }
LABEL_26:
        if ( v34 )
          v31 = v34;
      }
    }
LABEL_45:
    *(_DWORD *)(a1 + 64) = v30;
    if ( *v31 != -8 )
      --*(_DWORD *)(a1 + 68);
    *v31 = v24;
    *((_DWORD *)v31 + 2) = v18;
    goto LABEL_16;
  }
}
