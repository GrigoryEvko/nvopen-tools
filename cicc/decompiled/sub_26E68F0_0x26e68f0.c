// Function: sub_26E68F0
// Address: 0x26e68f0
//
__int64 __fastcall sub_26E68F0(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  __int64 (*v3)(); // rax
  int v4; // ecx
  unsigned int v5; // ebx
  int **v7; // rax
  __int64 v8; // r8
  int **v9; // r14
  int *v10; // r15
  int **v11; // rbx
  int *v12; // r13
  size_t v13; // r12
  unsigned __int64 v14; // rcx
  _QWORD *v15; // rax
  int v16; // r8d
  __int64 v17; // rcx
  int v18; // eax
  int v19; // edx
  int *v20; // rdi
  int v21; // r9d
  size_t v22; // rdx
  int v23; // r15d
  __int64 v24; // r14
  unsigned int v25; // r12d
  __int64 v26; // rbx
  const void *v27; // r13
  bool v28; // al
  unsigned int v29; // r12d
  int v30; // eax
  __int64 v31; // r11
  int v32; // r9d
  int v33; // r8d
  unsigned int i; // r10d
  const void *v35; // rsi
  bool v36; // al
  unsigned int v37; // r10d
  int v38; // eax
  int v39; // eax
  int v40; // eax
  int v41; // eax
  int v42; // r10d
  __int64 v43; // r8
  int v44; // r9d
  unsigned int v45; // r11d
  const void *v46; // rsi
  bool v47; // al
  unsigned int v48; // r11d
  int v49; // eax
  __int64 v50; // [rsp+8h] [rbp-158h]
  int v51; // [rsp+8h] [rbp-158h]
  int v52; // [rsp+8h] [rbp-158h]
  int v53; // [rsp+10h] [rbp-150h]
  size_t v54; // [rsp+10h] [rbp-150h]
  __int64 v55; // [rsp+10h] [rbp-150h]
  __int64 v56; // [rsp+18h] [rbp-148h]
  unsigned int v57; // [rsp+18h] [rbp-148h]
  __int64 v58; // [rsp+20h] [rbp-140h]
  unsigned int v59; // [rsp+20h] [rbp-140h]
  int v60; // [rsp+20h] [rbp-140h]
  int *v61; // [rsp+30h] [rbp-130h]
  int v62; // [rsp+30h] [rbp-130h]
  int v63; // [rsp+30h] [rbp-130h]
  __int64 v64; // [rsp+30h] [rbp-130h]
  int **v65; // [rsp+38h] [rbp-128h]
  __int64 v66; // [rsp+38h] [rbp-128h]
  int v67; // [rsp+38h] [rbp-128h]
  int *v68; // [rsp+48h] [rbp-118h]
  int v69; // [rsp+48h] [rbp-118h]
  int **v70; // [rsp+48h] [rbp-118h]
  __int64 v71; // [rsp+48h] [rbp-118h]
  _QWORD *v72; // [rsp+50h] [rbp-110h]
  unsigned __int64 v74; // [rsp+60h] [rbp-100h] BYREF
  __int64 v75; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v76; // [rsp+78h] [rbp-E8h]
  __int64 v77; // [rsp+80h] [rbp-E0h]
  __int64 v78; // [rsp+88h] [rbp-D8h]
  unsigned __int64 v79; // [rsp+90h] [rbp-D0h] BYREF

  v72 = a1 + 1;
  v2 = *(_DWORD *)(a2 + 16);
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  if ( v2 )
  {
    v7 = *(int ***)(a2 + 8);
    v8 = 2LL * *(unsigned int *)(a2 + 24);
    v9 = &v7[v8];
    if ( v7 != &v7[v8] )
    {
      while ( 1 )
      {
        v10 = *v7;
        v11 = v7;
        if ( *v7 != (int *)-1LL && v10 != (int *)-2LL )
          break;
        v7 += 2;
        if ( v9 == v7 )
          goto LABEL_2;
      }
      if ( v7 != v9 )
      {
        v12 = (int *)&v79;
        while ( 1 )
        {
          v13 = (size_t)v11[1];
          v68 = *v11;
          v14 = v13;
          if ( v10 )
          {
            sub_C7D030(v12);
            sub_C7D280(v12, v10, v13);
            sub_C7D290(v12, &v74);
            v14 = v74;
          }
          v79 = v14;
          v15 = sub_C1DD00(v72, v14 % a1[2], v12, v14);
          if ( v15 && *v15 )
            goto LABEL_17;
          v16 = v78;
          if ( !(_DWORD)v78 )
          {
            ++v75;
            goto LABEL_26;
          }
          v62 = v78;
          v66 = v76;
          v30 = sub_C94890(v68, v13);
          v31 = 0;
          v32 = v62 - 1;
          v33 = 1;
          for ( i = (v62 - 1) & v30; ; i = v32 & v37 )
          {
            v17 = v66 + 16LL * i;
            v35 = *(const void **)v17;
            v36 = (int *)((char *)v10 + 1) == 0;
            if ( *(_QWORD *)v17 != -1 )
            {
              v36 = (int *)((char *)v10 + 2) == 0;
              if ( v35 != (const void *)-2LL )
              {
                if ( v13 != *(_QWORD *)(v17 + 8) )
                  goto LABEL_36;
                v50 = v66 + 16LL * i;
                v53 = v33;
                v56 = v31;
                v59 = i;
                v63 = v32;
                if ( !v13 )
                  goto LABEL_17;
                v38 = memcmp(v10, v35, v13);
                v32 = v63;
                i = v59;
                v31 = v56;
                v33 = v53;
                v17 = v50;
                v36 = v38 == 0;
              }
            }
            if ( v36 )
              goto LABEL_17;
            if ( v35 == (const void *)-1LL )
              break;
LABEL_36:
            if ( v35 != (const void *)-2LL || v31 )
              v17 = v31;
            v37 = v33 + i;
            v31 = v17;
            ++v33;
          }
          v16 = v78;
          if ( v31 )
            v17 = v31;
          ++v75;
          v39 = v77 + 1;
          if ( 4 * ((int)v77 + 1) < (unsigned int)(3 * v78) )
          {
            if ( (int)v78 - (v39 + HIDWORD(v77)) > (unsigned int)v78 >> 3 )
              goto LABEL_47;
            sub_BA8070((__int64)&v75, v78);
            v17 = 0;
            v67 = v78;
            if ( !(_DWORD)v78 )
              goto LABEL_56;
            v71 = v76;
            v41 = sub_C94890(v10, v13);
            v42 = 1;
            v43 = 0;
            v44 = v67 - 1;
            v45 = (v67 - 1) & v41;
            while ( 2 )
            {
              v17 = v71 + 16LL * v45;
              v46 = *(const void **)v17;
              if ( *(_QWORD *)v17 == -1 )
              {
                if ( v10 == (int *)-1LL )
                  goto LABEL_56;
LABEL_69:
                if ( v43 )
                  v17 = v43;
                goto LABEL_56;
              }
              v47 = (int *)((char *)v10 + 2) == 0;
              if ( v46 != (const void *)-2LL )
              {
                if ( *(_QWORD *)(v17 + 8) != v13 )
                {
LABEL_62:
                  if ( v43 || v46 != (const void *)-2LL )
                    v17 = v43;
                  v48 = v42 + v45;
                  v43 = v17;
                  ++v42;
                  v45 = v44 & v48;
                  continue;
                }
                v52 = v42;
                v55 = v43;
                v57 = v45;
                v60 = v44;
                if ( !v13 )
                  goto LABEL_56;
                v64 = v71 + 16LL * v45;
                v49 = memcmp(v10, v46, v13);
                v17 = v64;
                v44 = v60;
                v45 = v57;
                v43 = v55;
                v42 = v52;
                v47 = v49 == 0;
              }
              break;
            }
            if ( v47 )
              goto LABEL_56;
            if ( v46 == (const void *)-1LL )
              goto LABEL_69;
            goto LABEL_62;
          }
LABEL_26:
          sub_BA8070((__int64)&v75, 2 * v16);
          v17 = 0;
          v69 = v78;
          if ( !(_DWORD)v78 )
            goto LABEL_56;
          v58 = v76;
          v18 = sub_C94890(v10, v13);
          v19 = v69;
          v20 = v10;
          v70 = v11;
          v21 = v19 - 1;
          v61 = v12;
          v22 = v13;
          v23 = 1;
          v65 = v9;
          v24 = 0;
          v25 = v21 & v18;
          while ( 2 )
          {
            v26 = v58 + 16LL * v25;
            v27 = *(const void **)v26;
            if ( *(_QWORD *)v26 != -1 )
            {
              v28 = (int *)((char *)v20 + 2) == 0;
              if ( v27 != (const void *)-2LL )
              {
                if ( *(_QWORD *)(v26 + 8) != v22 )
                {
LABEL_31:
                  v29 = v23 + v25;
                  ++v23;
                  v25 = v21 & v29;
                  continue;
                }
                v51 = v21;
                if ( !v22 )
                  goto LABEL_55;
                v54 = v22;
                v40 = memcmp(v20, v27, v22);
                v22 = v54;
                v21 = v51;
                v28 = v40 == 0;
              }
              if ( v28 )
              {
LABEL_55:
                v17 = v58 + 16LL * v25;
                v9 = v65;
                v10 = v20;
                v13 = v22;
                v11 = v70;
                v12 = v61;
                goto LABEL_56;
              }
              if ( !v24 && v27 == (const void *)-2LL )
                v24 = v58 + 16LL * v25;
              goto LABEL_31;
            }
            break;
          }
          v17 = v58 + 16LL * v25;
          v43 = v24;
          v11 = v70;
          v10 = v20;
          v9 = v65;
          v12 = v61;
          v13 = v22;
          if ( v20 != (int *)-1LL )
            goto LABEL_69;
LABEL_56:
          v39 = v77 + 1;
LABEL_47:
          LODWORD(v77) = v39;
          if ( *(_QWORD *)v17 != -1 )
            --HIDWORD(v77);
          *(_QWORD *)v17 = v10;
          *(_QWORD *)(v17 + 8) = v13;
LABEL_17:
          v11 += 2;
          if ( v11 != v9 )
          {
            while ( 1 )
            {
              v10 = *v11;
              if ( *v11 != (int *)-1LL && v10 != (int *)-2LL )
                break;
              v11 += 2;
              if ( v9 == v11 )
                goto LABEL_2;
            }
            if ( v11 != v9 )
              continue;
          }
          break;
        }
      }
    }
  }
LABEL_2:
  v3 = *(__int64 (**)())(*a1 + 80LL);
  if ( v3 == sub_C1E990 )
  {
    sub_C1AFD0();
    v4 = 9;
  }
  else
  {
    v4 = ((__int64 (__fastcall *)(_QWORD *, __int64 *, _QWORD *))v3)(a1, &v75, v72);
    if ( !v4 )
    {
      v5 = 0;
      sub_C1AFD0();
      goto LABEL_5;
    }
  }
  v5 = v4;
LABEL_5:
  sub_C7D6A0(v76, 16LL * (unsigned int)v78, 8);
  return v5;
}
