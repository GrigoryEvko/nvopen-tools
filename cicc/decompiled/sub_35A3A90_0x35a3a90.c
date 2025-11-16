// Function: sub_35A3A90
// Address: 0x35a3a90
//
__int64 __fastcall sub_35A3A90(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // rdx
  __int64 *v3; // r13
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // ebx
  __int64 v7; // rcx
  int v8; // r15d
  __int64 *v9; // r10
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _DWORD *v13; // rax
  __int64 *v14; // r14
  int v15; // eax
  __int64 v16; // rax
  __int64 *v17; // r15
  __int64 v19; // r12
  int v20; // eax
  int v21; // eax
  __int64 v22; // rbx
  __int64 v23; // r14
  __int64 v24; // rbx
  __int64 *v25; // r15
  __int64 v26; // r13
  int v27; // esi
  unsigned __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r12
  int v31; // eax
  int v32; // r14d
  int v33; // eax
  __int64 v34; // r8
  int v35; // eax
  unsigned int v36; // r8d
  int v37; // r14d
  __int64 v38; // rsi
  unsigned int v39; // r9d
  __int64 v40; // rdi
  __int64 *v41; // r10
  int v42; // ecx
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // r11
  unsigned int v46; // r10d
  int v47; // ecx
  __int64 v48; // rsi
  unsigned int v49; // edx
  __int64 v50; // rax
  __int64 v51; // r11
  unsigned int v52; // eax
  int v53; // r14d
  int v54; // edx
  unsigned int v55; // eax
  int v56; // eax
  int v57; // esi
  __int64 *v58; // [rsp+8h] [rbp-88h]
  unsigned int v59; // [rsp+10h] [rbp-80h]
  unsigned int v60; // [rsp+10h] [rbp-80h]
  int v61; // [rsp+14h] [rbp-7Ch]
  __int64 *v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int64 v64; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v65; // [rsp+38h] [rbp-58h] BYREF
  __int64 v66; // [rsp+40h] [rbp-50h] BYREF
  __int64 v67; // [rsp+48h] [rbp-48h]
  __int64 v68; // [rsp+50h] [rbp-40h]
  unsigned int v69; // [rsp+58h] [rbp-38h]

  v1 = *(_QWORD *)a1;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  *(_DWORD *)(a1 + 128) = 1;
  v2 = *(__int64 **)(v1 + 8);
  v66 = 0;
  if ( v2 == *(__int64 **)(v1 + 16) )
    return sub_C7D6A0(v67, 16LL * v69, 8);
  v3 = (__int64 *)a1;
  v4 = 0;
  v5 = 0;
  v6 = 0;
  v7 = 0;
  while ( 1 )
  {
    v14 = &v2[v7];
    if ( v4 )
    {
      v8 = 1;
      v9 = 0;
      v10 = (v4 - 1) & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      v11 = (_QWORD *)(v5 + 16LL * v10);
      v12 = *v11;
      if ( *v14 == *v11 )
      {
LABEL_4:
        v13 = v11 + 1;
        goto LABEL_5;
      }
      while ( v12 != -4096 )
      {
        if ( v12 == -8192 && !v9 )
          v9 = v11;
        v10 = (v4 - 1) & (v8 + v10);
        v11 = (_QWORD *)(v5 + 16LL * v10);
        v12 = *v11;
        if ( *v14 == *v11 )
          goto LABEL_4;
        ++v8;
      }
      if ( !v9 )
        v9 = v11;
      ++v66;
      v15 = v68 + 1;
      v65 = v9;
      if ( 4 * ((int)v68 + 1) < 3 * v4 )
      {
        if ( v4 - (v15 + HIDWORD(v68)) > v4 >> 3 )
          goto LABEL_21;
        goto LABEL_10;
      }
    }
    else
    {
      ++v66;
      v65 = 0;
    }
    v4 *= 2;
LABEL_10:
    sub_2E261E0((__int64)&v66, v4);
    sub_35472E0((__int64)&v66, v14, &v65);
    v9 = v65;
    v15 = v68 + 1;
LABEL_21:
    LODWORD(v68) = v15;
    if ( *v9 != -4096 )
      --HIDWORD(v68);
    v16 = *v14;
    *((_DWORD *)v9 + 2) = 0;
    *v9 = v16;
    v13 = v9 + 1;
LABEL_5:
    *v13 = v6++;
    v7 = v6;
    v2 = *(__int64 **)(*v3 + 8);
    if ( v6 >= (unsigned __int64)((__int64)(*(_QWORD *)(*v3 + 16) - (_QWORD)v2) >> 3) )
      break;
    v5 = v67;
    v4 = v69;
  }
  v62 = *(__int64 **)(*v3 + 16);
  if ( v62 != v2 )
  {
    v17 = *(__int64 **)(*v3 + 8);
    while ( 1 )
    {
      v19 = *v17;
      v20 = *(unsigned __int16 *)(*v17 + 68);
      v63 = *v17;
      if ( v20 != 68 )
      {
        if ( v20 )
        {
          v21 = sub_3598DB0(*v3, v19);
          v22 = *(_QWORD *)(v19 + 32);
          v61 = v21;
          v23 = v22 + 40LL * (*(_DWORD *)(v19 + 40) & 0xFFFFFF);
          v24 = v22 + 40LL * (unsigned int)sub_2E88FE0(v19);
          if ( v23 != v24 )
            break;
        }
      }
LABEL_25:
      if ( v62 == ++v17 )
        return sub_C7D6A0(v67, 16LL * v69, 8);
    }
    v58 = v17;
    v25 = v3;
    v26 = v23;
    while ( 2 )
    {
      if ( *(_BYTE *)v24 )
        goto LABEL_33;
      v27 = *(_DWORD *)(v24 + 8);
      if ( v27 >= 0 )
        goto LABEL_33;
      v28 = sub_2EBEE10(v25[3], v27);
      v29 = v25[6];
      v64 = v28;
      v30 = v28;
      if ( v29 != *(_QWORD *)(v28 + 24) )
        goto LABEL_33;
      v31 = *(unsigned __int16 *)(v28 + 68);
      if ( !v31 || (v32 = 1, v31 == 68) )
      {
        v32 = 2;
        v33 = sub_3598190(v30, v29);
        v64 = sub_2EBEE10(v34, v33);
        v30 = v64;
      }
      v35 = sub_3598DB0(*v25, v30);
      v36 = v69;
      v37 = v61 - v35 + v32;
      if ( v69 )
      {
        v38 = v63;
        v39 = v69 - 1;
        v40 = v67;
        v41 = 0;
        v42 = 1;
        v43 = (v69 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
        v44 = (__int64 *)(v67 + 16LL * v43);
        v45 = *v44;
        if ( v63 == *v44 )
        {
LABEL_42:
          v46 = *((_DWORD *)v44 + 2);
          goto LABEL_43;
        }
        while ( v45 != -4096 )
        {
          if ( v45 == -8192 && !v41 )
            v41 = v44;
          v43 = v39 & (v42 + v43);
          v44 = (__int64 *)(v67 + 16LL * v43);
          v45 = *v44;
          if ( v63 == *v44 )
            goto LABEL_42;
          ++v42;
        }
        if ( !v41 )
          v41 = v44;
        ++v66;
        v54 = v68 + 1;
        v65 = v41;
        if ( 4 * ((int)v68 + 1) < 3 * v69 )
        {
          if ( v69 - HIDWORD(v68) - v54 > v69 >> 3 )
            goto LABEL_58;
          v57 = v69;
LABEL_80:
          sub_2E261E0((__int64)&v66, v57);
          sub_35472E0((__int64)&v66, &v63, &v65);
          v38 = v63;
          v41 = v65;
          v54 = v68 + 1;
LABEL_58:
          LODWORD(v68) = v54;
          if ( *v41 != -4096 )
            --HIDWORD(v68);
          *v41 = v38;
          *((_DWORD *)v41 + 2) = 0;
          v36 = v69;
          if ( !v69 )
          {
            ++v66;
            v55 = 0;
            v65 = 0;
            goto LABEL_62;
          }
          v40 = v67;
          v30 = v64;
          v39 = v69 - 1;
          v46 = 0;
LABEL_43:
          v47 = 1;
          v48 = 0;
          v49 = v39 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
          v50 = v40 + 16LL * v49;
          v51 = *(_QWORD *)v50;
          if ( *(_QWORD *)v50 == v30 )
          {
LABEL_44:
            v52 = *(_DWORD *)(v50 + 8);
          }
          else
          {
            while ( v51 != -4096 )
            {
              if ( !v48 && v51 == -8192 )
                v48 = v50;
              v49 = v39 & (v47 + v49);
              v50 = v40 + 16LL * v49;
              v51 = *(_QWORD *)v50;
              if ( *(_QWORD *)v50 == v30 )
                goto LABEL_44;
              ++v47;
            }
            if ( !v48 )
              v48 = v50;
            ++v66;
            v56 = v68 + 1;
            v65 = (__int64 *)v48;
            if ( 4 * ((int)v68 + 1) >= 3 * v36 )
            {
              v55 = v36;
              v36 = v46;
LABEL_62:
              v59 = v36;
              sub_2E261E0((__int64)&v66, 2 * v55);
              sub_35472E0((__int64)&v66, (__int64 *)&v64, &v65);
              v30 = v64;
              v48 = (__int64)v65;
              v56 = v68 + 1;
              v46 = v59;
            }
            else if ( v36 - (v56 + HIDWORD(v68)) <= v36 >> 3 )
            {
              v60 = v46;
              sub_2E261E0((__int64)&v66, v36);
              sub_35472E0((__int64)&v66, (__int64 *)&v64, &v65);
              v30 = v64;
              v48 = (__int64)v65;
              v46 = v60;
              v56 = v68 + 1;
            }
            LODWORD(v68) = v56;
            if ( *(_QWORD *)v48 != -4096 )
              --HIDWORD(v68);
            *(_QWORD *)v48 = v30;
            v52 = 0;
            *(_DWORD *)(v48 + 8) = 0;
          }
          v53 = (v52 < v46) + v37 - 1;
          if ( *((_DWORD *)v25 + 32) >= v53 )
            v53 = *((_DWORD *)v25 + 32);
          *((_DWORD *)v25 + 32) = v53;
LABEL_33:
          v24 += 40;
          if ( v26 == v24 )
          {
            v3 = v25;
            v17 = v58;
            goto LABEL_25;
          }
          continue;
        }
      }
      else
      {
        ++v66;
        v65 = 0;
      }
      break;
    }
    v57 = 2 * v69;
    goto LABEL_80;
  }
  return sub_C7D6A0(v67, 16LL * v69, 8);
}
