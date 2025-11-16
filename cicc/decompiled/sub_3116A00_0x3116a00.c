// Function: sub_3116A00
// Address: 0x3116a00
//
__int64 __fastcall sub_3116A00(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // esi
  _DWORD *v4; // rax
  int v5; // ebx
  _DWORD *v6; // rdx
  int v7; // r10d
  unsigned int v8; // edi
  _DWORD *v9; // rax
  int v10; // ecx
  __int64 v11; // r13
  int v12; // eax
  unsigned int *v13; // r14
  unsigned int v14; // r12d
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  unsigned int *v18; // rdx
  int v19; // r10d
  unsigned int v20; // edi
  _DWORD *v21; // rax
  int v22; // ecx
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdi
  _QWORD *v29; // r8
  _QWORD *v30; // rax
  _QWORD *v31; // rsi
  unsigned __int64 *v32; // r12
  unsigned __int64 v33; // r15
  unsigned __int64 v34; // rdi
  int v36; // eax
  unsigned __int64 *v37; // rax
  unsigned __int64 *v38; // r12
  __int64 v39; // rsi
  char v40; // al
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // r8
  __int64 v43; // r10
  unsigned __int64 *v44; // rax
  void *v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // r10
  _QWORD *v48; // r9
  _QWORD *v49; // rsi
  unsigned __int64 v50; // rdi
  _QWORD *v51; // rcx
  unsigned __int64 v52; // rdx
  _QWORD **v53; // rax
  unsigned __int64 v54; // rdi
  __int64 v55; // rcx
  unsigned int v56; // esi
  int v57; // r11d
  unsigned int *v58; // r10
  unsigned int *v59; // r9
  __int64 v60; // r15
  int v61; // r10d
  unsigned int v62; // ecx
  int v63; // ecx
  unsigned int v64; // eax
  int v65; // r11d
  int v66; // edi
  _DWORD *v67; // rsi
  int v68; // esi
  unsigned int v69; // r12d
  _DWORD *v70; // rax
  int v71; // edi
  int v72; // r9d
  _DWORD *v73; // rdx
  unsigned int v74; // ecx
  unsigned __int64 v75; // [rsp+8h] [rbp-88h]
  unsigned __int64 v76; // [rsp+8h] [rbp-88h]
  __int64 v77; // [rsp+10h] [rbp-80h]
  _QWORD *v78; // [rsp+18h] [rbp-78h]
  __int64 n; // [rsp+20h] [rbp-70h]
  size_t na; // [rsp+20h] [rbp-70h]
  unsigned int *v82; // [rsp+30h] [rbp-60h]
  __int64 v83; // [rsp+38h] [rbp-58h]
  __int64 v84; // [rsp+40h] [rbp-50h] BYREF
  _DWORD *v85; // [rsp+48h] [rbp-48h]
  __int64 v86; // [rsp+50h] [rbp-40h]
  unsigned int v87; // [rsp+58h] [rbp-38h]

  v2 = *a1;
  v84 = 1;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  sub_3116820((__int64)&v84, 0);
  if ( !v87 )
  {
LABEL_147:
    LODWORD(v86) = v86 + 1;
    BUG();
  }
  v3 = *v85;
  v4 = v85;
  if ( *v85 )
  {
    v72 = 1;
    v73 = v85;
    v4 = 0;
    v74 = 0;
    while ( v3 != -1 )
    {
      if ( !v4 && v3 == -2 )
        v4 = v73;
      v74 = (v87 - 1) & (v72 + v74);
      v73 = &v85[4 * v74];
      v3 = *v73;
      if ( !*v73 )
      {
        v4 = &v85[4 * v74];
        goto LABEL_3;
      }
      ++v72;
    }
    if ( !v4 )
      v4 = v73;
  }
LABEL_3:
  LODWORD(v86) = v86 + 1;
  if ( *v4 != -1 )
    --HIDWORD(v86);
  *((_QWORD *)v4 + 1) = v2;
  *v4 = 0;
  v83 = a2 + 8;
  v77 = *(_QWORD *)(a2 + 24);
  if ( v77 != a2 + 8 )
  {
    do
    {
      v5 = *(_DWORD *)(v77 + 32);
      if ( v87 )
      {
        v6 = 0;
        v7 = 1;
        v8 = (v87 - 1) & (37 * v5);
        v9 = &v85[4 * v8];
        v10 = *v9;
        if ( v5 == *v9 )
        {
LABEL_8:
          v11 = *((_QWORD *)v9 + 1);
          goto LABEL_9;
        }
        while ( v10 != -1 )
        {
          if ( v10 == -2 && !v6 )
            v6 = v9;
          v8 = (v87 - 1) & (v7 + v8);
          v9 = &v85[4 * v8];
          v10 = *v9;
          if ( v5 == *v9 )
            goto LABEL_8;
          ++v7;
        }
        if ( !v6 )
          v6 = v9;
        ++v84;
        v63 = v86 + 1;
        if ( 4 * ((int)v86 + 1) < 3 * v87 )
        {
          if ( v87 - HIDWORD(v86) - v63 <= v87 >> 3 )
          {
            sub_3116820((__int64)&v84, v87);
            if ( !v87 )
              goto LABEL_147;
            v68 = 1;
            v69 = (v87 - 1) & (37 * v5);
            v63 = v86 + 1;
            v70 = 0;
            v6 = &v85[4 * v69];
            v71 = *v6;
            if ( v5 != *v6 )
            {
              while ( v71 != -1 )
              {
                if ( !v70 && v71 == -2 )
                  v70 = v6;
                v69 = (v87 - 1) & (v68 + v69);
                v6 = &v85[4 * v69];
                v71 = *v6;
                if ( v5 == *v6 )
                  goto LABEL_98;
                ++v68;
              }
              if ( v70 )
                v6 = v70;
            }
          }
          goto LABEL_98;
        }
      }
      else
      {
        ++v84;
      }
      sub_3116820((__int64)&v84, 2 * v87);
      if ( !v87 )
        goto LABEL_147;
      v64 = (v87 - 1) & (37 * v5);
      v63 = v86 + 1;
      v6 = &v85[4 * v64];
      v65 = *v6;
      if ( v5 != *v6 )
      {
        v66 = 1;
        v67 = 0;
        while ( v65 != -1 )
        {
          if ( v65 == -2 && !v67 )
            v67 = v6;
          v64 = (v87 - 1) & (v66 + v64);
          v6 = &v85[4 * v64];
          v65 = *v6;
          if ( v5 == *v6 )
            goto LABEL_98;
          ++v66;
        }
        if ( v67 )
          v6 = v67;
      }
LABEL_98:
      LODWORD(v86) = v63;
      if ( *v6 != -1 )
        --HIDWORD(v86);
      *v6 = v5;
      v11 = 0;
      *((_QWORD *)v6 + 1) = 0;
LABEL_9:
      *(_QWORD *)v11 = *(_QWORD *)(v77 + 40);
      v12 = *(_DWORD *)(v77 + 48);
      if ( v12 )
      {
        *(_DWORD *)(v11 + 8) = v12;
        *(_BYTE *)(v11 + 12) = 1;
      }
      v13 = *(unsigned int **)(v77 + 56);
      v78 = (_QWORD *)(v11 + 32);
      v82 = *(unsigned int **)(v77 + 64);
      if ( v82 != v13 )
      {
        while ( 1 )
        {
          v14 = *v13;
          v15 = sub_22077B0(0x48u);
          v16 = v15;
          if ( v15 )
          {
            *(_QWORD *)(v15 + 64) = 0;
            v17 = v15 + 64;
            *(_OWORD *)(v17 - 64) = 0;
            *(_OWORD *)(v17 - 32) = 0;
            *(_OWORD *)(v17 - 16) = 0;
            *(_QWORD *)(v16 + 16) = v17;
            *(_QWORD *)(v16 + 24) = 1;
            *(_QWORD *)(v16 + 56) = 0;
            *(_DWORD *)(v16 + 48) = 1065353216;
          }
          if ( !v87 )
            break;
          v18 = 0;
          v19 = 1;
          v20 = (v87 - 1) & (37 * v14);
          v21 = &v85[4 * v20];
          v22 = *v21;
          if ( v14 == *v21 )
          {
LABEL_16:
            *((_QWORD *)v21 + 1) = v16;
            v23 = *(_QWORD *)(a2 + 16);
            if ( !v23 )
              goto LABEL_49;
            goto LABEL_17;
          }
          while ( v22 != -1 )
          {
            if ( v22 == -2 && !v18 )
              v18 = v21;
            v20 = (v87 - 1) & (v19 + v20);
            v21 = &v85[4 * v20];
            v22 = *v21;
            if ( v14 == *v21 )
              goto LABEL_16;
            ++v19;
          }
          if ( !v18 )
            v18 = v21;
          ++v84;
          v36 = v86 + 1;
          if ( 4 * ((int)v86 + 1) >= 3 * v87 )
            goto LABEL_74;
          if ( v87 - HIDWORD(v86) - v36 <= v87 >> 3 )
          {
            sub_3116820((__int64)&v84, v87);
            if ( !v87 )
              goto LABEL_147;
            v59 = 0;
            LODWORD(v60) = (v87 - 1) & (37 * v14);
            v61 = 1;
            v36 = v86 + 1;
            v18 = &v85[4 * (unsigned int)v60];
            v62 = *v18;
            if ( v14 != *v18 )
            {
              while ( v62 != -1 )
              {
                if ( !v59 && v62 == -2 )
                  v59 = v18;
                v60 = (v87 - 1) & ((_DWORD)v60 + v61);
                v18 = &v85[4 * v60];
                v62 = *v18;
                if ( v14 == *v18 )
                  goto LABEL_46;
                ++v61;
              }
              if ( v59 )
                v18 = v59;
            }
          }
LABEL_46:
          LODWORD(v86) = v36;
          if ( *v18 != -1 )
            --HIDWORD(v86);
          *v18 = v14;
          *((_QWORD *)v18 + 1) = 0;
          *((_QWORD *)v18 + 1) = v16;
          v23 = *(_QWORD *)(a2 + 16);
          if ( !v23 )
            goto LABEL_49;
LABEL_17:
          v24 = a2 + 8;
          do
          {
            while ( 1 )
            {
              v25 = *(_QWORD *)(v23 + 16);
              v26 = *(_QWORD *)(v23 + 24);
              if ( v14 <= *(_DWORD *)(v23 + 32) )
                break;
              v23 = *(_QWORD *)(v23 + 24);
              if ( !v26 )
                goto LABEL_21;
            }
            v24 = v23;
            v23 = *(_QWORD *)(v23 + 16);
          }
          while ( v25 );
LABEL_21:
          if ( v83 == v24 || v14 < *(_DWORD *)(v24 + 32) )
LABEL_49:
            sub_426320((__int64)"map::at");
          v27 = *(_QWORD *)(v24 + 40);
          v28 = *(_QWORD *)(v11 + 24);
          v29 = *(_QWORD **)(*(_QWORD *)(v11 + 16) + 8 * (v27 % v28));
          if ( !v29 )
            goto LABEL_50;
          v30 = (_QWORD *)*v29;
          if ( v27 != *(_QWORD *)(*v29 + 8LL) )
          {
            while ( 1 )
            {
              v31 = (_QWORD *)*v30;
              if ( !*v30 )
                break;
              v29 = v30;
              if ( v27 % v28 != v31[1] % v28 )
                break;
              v30 = (_QWORD *)*v30;
              if ( v27 == v31[1] )
                goto LABEL_28;
            }
LABEL_50:
            v37 = (unsigned __int64 *)sub_22077B0(0x18u);
            v38 = v37;
            if ( v37 )
              *v37 = 0;
            v37[1] = v27;
            v37[2] = 0;
            v39 = *(_QWORD *)(v11 + 24);
            v40 = sub_222DA10(v11 + 48, v39, *(_QWORD *)(v11 + 40), 1);
            v42 = v41;
            if ( !v40 )
            {
              v43 = 8 * (v27 % v28);
              v44 = *(unsigned __int64 **)(*(_QWORD *)(v11 + 16) + v43);
              if ( v44 )
                goto LABEL_54;
LABEL_69:
              *v38 = *(_QWORD *)(v11 + 32);
              *(_QWORD *)(v11 + 32) = v38;
              if ( *v38 )
                *(_QWORD *)(*(_QWORD *)(v11 + 16) + 8LL * (*(_QWORD *)(*v38 + 8) % *(_QWORD *)(v11 + 24))) = v38;
              *(_QWORD *)(*(_QWORD *)(v11 + 16) + v43) = v78;
              goto LABEL_55;
            }
            if ( v41 == 1 )
            {
              v48 = (_QWORD *)(v11 + 64);
              *(_QWORD *)(v11 + 64) = 0;
              v47 = v11 + 64;
            }
            else
            {
              if ( v41 > 0xFFFFFFFFFFFFFFFLL )
                sub_4261EA(v11 + 48, v39, v41);
              v75 = v41;
              n = 8 * v41;
              v45 = (void *)sub_22077B0(8 * v41);
              v46 = memset(v45, 0, n);
              v42 = v75;
              v47 = v11 + 64;
              v48 = v46;
            }
            v49 = *(_QWORD **)(v11 + 32);
            *(_QWORD *)(v11 + 32) = 0;
            if ( !v49 )
            {
LABEL_66:
              v54 = *(_QWORD *)(v11 + 16);
              if ( v47 != v54 )
              {
                v76 = v42;
                na = (size_t)v48;
                j_j___libc_free_0(v54);
                v42 = v76;
                v48 = (_QWORD *)na;
              }
              *(_QWORD *)(v11 + 24) = v42;
              *(_QWORD *)(v11 + 16) = v48;
              v43 = 8 * (v27 % v42);
              v44 = (unsigned __int64 *)v48[(unsigned __int64)v43 / 8];
              if ( !v44 )
                goto LABEL_69;
LABEL_54:
              *v38 = *v44;
              **(_QWORD **)(*(_QWORD *)(v11 + 16) + v43) = v38;
LABEL_55:
              ++*(_QWORD *)(v11 + 40);
              v32 = v38 + 2;
              goto LABEL_29;
            }
            v50 = 0;
            while ( 1 )
            {
              v51 = v49;
              v49 = (_QWORD *)*v49;
              v52 = v51[1] % v42;
              v53 = (_QWORD **)&v48[v52];
              if ( *v53 )
                break;
              *v51 = *(_QWORD *)(v11 + 32);
              *(_QWORD *)(v11 + 32) = v51;
              *v53 = v78;
              if ( !*v51 )
              {
                v50 = v52;
LABEL_62:
                if ( !v49 )
                  goto LABEL_66;
                continue;
              }
              v48[v50] = v51;
              v50 = v52;
              if ( !v49 )
                goto LABEL_66;
            }
            *v51 = **v53;
            **v53 = v51;
            goto LABEL_62;
          }
LABEL_28:
          v32 = (unsigned __int64 *)(*v29 + 16LL);
          if ( !*v29 )
            goto LABEL_50;
LABEL_29:
          v33 = *v32;
          *v32 = v16;
          if ( v33 )
          {
            sub_3112140(v33 + 16);
            v34 = *(_QWORD *)(v33 + 16);
            if ( v34 != v33 + 64 )
              j_j___libc_free_0(v34);
            j_j___libc_free_0(v33);
          }
          if ( v82 == ++v13 )
            goto LABEL_34;
        }
        ++v84;
LABEL_74:
        sub_3116820((__int64)&v84, 2 * v87);
        if ( !v87 )
          goto LABEL_147;
        LODWORD(v55) = (v87 - 1) & (37 * v14);
        v36 = v86 + 1;
        v18 = &v85[4 * (unsigned int)v55];
        v56 = *v18;
        if ( v14 != *v18 )
        {
          v57 = 1;
          v58 = 0;
          while ( v56 != -1 )
          {
            if ( v56 == -2 && !v58 )
              v58 = v18;
            v55 = (v87 - 1) & ((_DWORD)v55 + v57);
            v18 = &v85[4 * v55];
            v56 = *v18;
            if ( v14 == *v18 )
              goto LABEL_46;
            ++v57;
          }
          if ( v58 )
            v18 = v58;
        }
        goto LABEL_46;
      }
LABEL_34:
      v77 = sub_220EF30(v77);
    }
    while ( v83 != v77 );
  }
  return sub_C7D6A0((__int64)v85, 16LL * v87, 8);
}
