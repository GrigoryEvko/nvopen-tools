// Function: sub_354FCF0
// Address: 0x354fcf0
//
__int64 __fastcall sub_354FCF0(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // eax
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // r13
  __int64 result; // rax
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 *v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rdi
  int v19; // eax
  unsigned __int64 v20; // rax
  int v21; // esi
  __int64 v22; // rdi
  unsigned __int64 v23; // rdx
  int v24; // esi
  unsigned int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // r9
  unsigned __int64 v28; // rsi
  int v29; // eax
  _QWORD *v30; // rbx
  int v31; // r14d
  _QWORD *v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // r9
  __int64 v35; // rdx
  int v36; // eax
  int v37; // r10d
  int v38; // r11d
  __int64 v39; // rcx
  int v40; // r9d
  _QWORD *v41; // rsi
  __int64 v42; // rdx
  int v43; // ebx
  _QWORD *v44; // rax
  int v45; // r14d
  unsigned int v46; // esi
  __int64 v47; // rdi
  int v48; // ebx
  __int64 *v49; // rdx
  __int64 v50; // r8
  unsigned int v51; // ecx
  _QWORD *v52; // rax
  _QWORD *v53; // r10
  unsigned __int64 *v54; // rax
  unsigned int v55; // esi
  __int64 v56; // rbx
  __int64 v57; // rdi
  int v58; // r14d
  __int64 *v59; // rdx
  __int64 v60; // r8
  unsigned int v61; // ecx
  __int64 *v62; // rax
  __int64 v63; // r10
  int v64; // ecx
  int v65; // r11d
  int v66; // eax
  int v67; // r10d
  int v68; // eax
  int v69; // r10d
  int v70; // eax
  int v71; // ecx
  int v72; // eax
  int v73; // ecx
  __int64 v74; // [rsp+8h] [rbp-88h]
  int v75; // [rsp+14h] [rbp-7Ch]
  int v76; // [rsp+20h] [rbp-70h]
  int v77; // [rsp+28h] [rbp-68h]
  int v78; // [rsp+28h] [rbp-68h]
  int v79; // [rsp+28h] [rbp-68h]
  __int64 v80; // [rsp+30h] [rbp-60h]
  __int64 v81; // [rsp+38h] [rbp-58h] BYREF
  unsigned int v82; // [rsp+48h] [rbp-48h] BYREF
  unsigned int v83; // [rsp+4Ch] [rbp-44h] BYREF
  _QWORD *v84; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v85[7]; // [rsp+58h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a1 + 960);
  v81 = a2;
  v7 = *(_QWORD *)(a1 + 944);
  if ( v6 )
  {
    v8 = v6 - 1;
    v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_3:
      v12 = v10[1];
      goto LABEL_4;
    }
    v19 = 1;
    while ( v11 != -4096 )
    {
      v67 = v19 + 1;
      v9 = v8 & (v19 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v19 = v67;
    }
  }
  v12 = 0;
LABEL_4:
  result = *(unsigned int *)(a1 + 4040);
  v14 = *(_QWORD *)(a1 + 4024);
  if ( !(_DWORD)result )
    return result;
  v15 = (result - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v16 = (__int64 *)(v14 + 24LL * v15);
  v17 = *v16;
  if ( v12 != *v16 )
  {
    v64 = 1;
    while ( v17 != -4096 )
    {
      v65 = v64 + 1;
      v15 = (result - 1) & (v64 + v15);
      v16 = (__int64 *)(v14 + 24LL * v15);
      v17 = *v16;
      if ( v12 == *v16 )
        goto LABEL_6;
      v64 = v65;
    }
    return result;
  }
LABEL_6:
  result = v14 + 24 * result;
  if ( v16 == (__int64 *)result )
    return result;
  v18 = *(_QWORD *)(a1 + 16);
  v80 = v16[1];
  result = *(_QWORD *)(*(_QWORD *)v18 + 824LL);
  if ( (__int64 (*)())result == sub_2FDC6B0 )
    return result;
  v74 = v16[2];
  result = ((__int64 (__fastcall *)(__int64, __int64, unsigned int *, unsigned int *))result)(v18, a2, &v82, &v83);
  if ( !(_BYTE)result )
    return result;
  v20 = sub_3544200(a1, *(_DWORD *)(*(_QWORD *)(v81 + 32) + 40LL * v82 + 8));
  v21 = *(_DWORD *)(a1 + 960);
  v22 = *(_QWORD *)(a1 + 944);
  v23 = v20;
  if ( v21 )
  {
    v24 = v21 - 1;
    v25 = v24 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v26 = (__int64 *)(v22 + 16LL * v25);
    v27 = *v26;
    if ( v23 == *v26 )
    {
LABEL_15:
      v28 = v26[1];
      goto LABEL_16;
    }
    v68 = 1;
    while ( v27 != -4096 )
    {
      v69 = v68 + 1;
      v25 = v24 & (v68 + v25);
      v26 = (__int64 *)(v22 + 16LL * v25);
      v27 = *v26;
      if ( v23 == *v26 )
        goto LABEL_15;
      v68 = v69;
    }
  }
  v28 = 0;
LABEL_16:
  v29 = sub_3542500(a3, v28);
  v30 = *(_QWORD **)(a3 + 48);
  v31 = v29;
  if ( v30 )
  {
    v32 = (_QWORD *)(a3 + 40);
    v33 = *(_QWORD **)(a3 + 48);
    do
    {
      while ( 1 )
      {
        v34 = v33[2];
        v35 = v33[3];
        if ( v33[4] >= v28 )
          break;
        v33 = (_QWORD *)v33[3];
        if ( !v35 )
          goto LABEL_21;
      }
      v32 = v33;
      v33 = (_QWORD *)v33[2];
    }
    while ( v34 );
LABEL_21:
    if ( (_QWORD *)(a3 + 40) == v32 )
    {
      v79 = *(_DWORD *)(a3 + 88);
      v66 = sub_3542500(a3, v12);
      v39 = a3 + 40;
      v38 = *(_DWORD *)(a3 + 80);
      v40 = v66;
      v37 = v79;
      v75 = 0 % v79;
    }
    else
    {
      if ( v32[4] > v28 )
        v32 = (_QWORD *)(a3 + 40);
      v76 = *(_DWORD *)(a3 + 80);
      v77 = *(_DWORD *)(a3 + 88);
      v75 = (*((_DWORD *)v32 + 10) - v76) % v77;
      v36 = sub_3542500(a3, v12);
      v37 = v77;
      v38 = v76;
      v39 = a3 + 40;
      v40 = v36;
    }
    v41 = (_QWORD *)v39;
    do
    {
      while ( 1 )
      {
        v42 = v30[2];
        result = v30[3];
        if ( v30[4] >= v12 )
          break;
        v30 = (_QWORD *)v30[3];
        if ( !result )
          goto LABEL_29;
      }
      v41 = v30;
      v30 = (_QWORD *)v30[2];
    }
    while ( v42 );
LABEL_29:
    if ( (_QWORD *)v39 == v41 )
    {
      result = (unsigned int)((*(_DWORD *)(v39 + 40) - v38) / v37);
      v43 = (*(_DWORD *)(v39 + 40) - v38) % v37;
    }
    else
    {
      v43 = 0;
      if ( v41[4] <= v12 )
      {
        result = (unsigned int)((*((_DWORD *)v41 + 10) - v38) / v37);
        v43 = (*((_DWORD *)v41 + 10) - v38) % v37;
      }
    }
  }
  else
  {
    v75 = 0;
    v43 = 0;
    result = sub_3542500(a3, v12);
    v40 = result;
  }
  if ( v31 > v40 )
  {
    v78 = v40;
    v44 = sub_2E7B2C0(*(_QWORD **)(a1 + 32), v81);
    v84 = v44;
    v45 = v31 - v78;
    if ( v43 > v75 )
    {
      --v45;
      sub_2EAB0C0(v44[4] + 40LL * v82, v80);
      v44 = v84;
    }
    *(_QWORD *)(v44[4] + 40LL * v83 + 24) = *(_QWORD *)(*(_QWORD *)(v81 + 32) + 40LL * v83 + 24) + v74 * v45;
    *(_BYTE *)(v12 + 254) |= 8u;
    *(_QWORD *)v12 = v44;
    v46 = *(_DWORD *)(a1 + 960);
    if ( v46 )
    {
      v47 = (__int64)v84;
      v48 = 1;
      v49 = 0;
      v50 = *(_QWORD *)(a1 + 944);
      v51 = (v46 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
      v52 = (_QWORD *)(v50 + 16LL * v51);
      v53 = (_QWORD *)*v52;
      if ( v84 == (_QWORD *)*v52 )
      {
LABEL_37:
        v54 = v52 + 1;
        goto LABEL_38;
      }
      while ( v53 != (_QWORD *)-4096LL )
      {
        if ( v53 == (_QWORD *)-8192LL && !v49 )
          v49 = v52;
        v51 = (v46 - 1) & (v48 + v51);
        v52 = (_QWORD *)(v50 + 16LL * v51);
        v53 = (_QWORD *)*v52;
        if ( v84 == (_QWORD *)*v52 )
          goto LABEL_37;
        ++v48;
      }
      if ( !v49 )
        v49 = v52;
      v72 = *(_DWORD *)(a1 + 952);
      ++*(_QWORD *)(a1 + 936);
      v73 = v72 + 1;
      v85[0] = v49;
      if ( 4 * (v72 + 1) < 3 * v46 )
      {
        if ( v46 - *(_DWORD *)(a1 + 956) - v73 > v46 >> 3 )
        {
LABEL_82:
          *(_DWORD *)(a1 + 952) = v73;
          if ( *v49 != -4096 )
            --*(_DWORD *)(a1 + 956);
          *v49 = v47;
          v54 = (unsigned __int64 *)(v49 + 1);
          v49[1] = 0;
LABEL_38:
          *v54 = v12;
          v55 = *(_DWORD *)(a1 + 4072);
          v56 = (__int64)v84;
          if ( v55 )
          {
            v57 = v81;
            v58 = 1;
            v59 = 0;
            v60 = *(_QWORD *)(a1 + 4056);
            v61 = (v55 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
            v62 = (__int64 *)(v60 + 16LL * v61);
            v63 = *v62;
            if ( *v62 == v81 )
            {
LABEL_40:
              result = (__int64)(v62 + 1);
LABEL_41:
              *(_QWORD *)result = v56;
              return result;
            }
            while ( v63 != -4096 )
            {
              if ( v63 == -8192 && !v59 )
                v59 = v62;
              v61 = (v55 - 1) & (v58 + v61);
              v62 = (__int64 *)(v60 + 16LL * v61);
              v63 = *v62;
              if ( v81 == *v62 )
                goto LABEL_40;
              ++v58;
            }
            if ( !v59 )
              v59 = v62;
            v70 = *(_DWORD *)(a1 + 4064);
            ++*(_QWORD *)(a1 + 4048);
            v71 = v70 + 1;
            v85[0] = v59;
            if ( 4 * (v70 + 1) < 3 * v55 )
            {
              if ( v55 - *(_DWORD *)(a1 + 4068) - v71 > v55 >> 3 )
              {
LABEL_66:
                *(_DWORD *)(a1 + 4064) = v71;
                if ( *v59 != -4096 )
                  --*(_DWORD *)(a1 + 4068);
                *v59 = v57;
                result = (__int64)(v59 + 1);
                v59[1] = 0;
                goto LABEL_41;
              }
LABEL_71:
              sub_2E48800(a1 + 4048, v55);
              sub_3547B30(a1 + 4048, &v81, v85);
              v57 = v81;
              v59 = (__int64 *)v85[0];
              v71 = *(_DWORD *)(a1 + 4064) + 1;
              goto LABEL_66;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 4048);
            v85[0] = 0;
          }
          v55 *= 2;
          goto LABEL_71;
        }
LABEL_87:
        sub_2F960A0(a1 + 936, v46);
        sub_3547A70(a1 + 936, (__int64 *)&v84, v85);
        v47 = (__int64)v84;
        v49 = (__int64 *)v85[0];
        v73 = *(_DWORD *)(a1 + 952) + 1;
        goto LABEL_82;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 936);
      v85[0] = 0;
    }
    v46 *= 2;
    goto LABEL_87;
  }
  return result;
}
