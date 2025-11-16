// Function: sub_2E271D0
// Address: 0x2e271d0
//
__int64 __fastcall sub_2E271D0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int16 *v7; // rax
  __int16 *v8; // r12
  unsigned int v9; // r14d
  __int64 v11; // r15
  unsigned int v12; // edx
  unsigned int v13; // r11d
  __int64 v14; // rbx
  unsigned int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // eax
  int v19; // eax
  unsigned __int64 v20; // rdx
  unsigned int *v21; // r13
  unsigned int v22; // ebx
  unsigned int *v23; // rax
  _BYTE *v24; // r13
  _BYTE *v25; // r12
  _BYTE *v26; // rbx
  __int64 v27; // r14
  unsigned int v28; // r13d
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // r9
  __int16 *v33; // rax
  _WORD *v34; // rdx
  __int16 v35; // cx
  unsigned __int16 v36; // ax
  _BYTE *v37; // r15
  unsigned int v38; // r15d
  bool v39; // zf
  __int16 *v40; // r12
  unsigned __int64 v41; // rdx
  unsigned int *v42; // rbx
  unsigned int *v43; // rax
  int v44; // eax
  unsigned __int64 v45; // rdx
  unsigned int *v46; // r12
  bool v47; // r9
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  int v52; // eax
  int v53; // edx
  int v54; // eax
  int v55; // esi
  unsigned int v56; // eax
  __int64 v57; // rdi
  int v58; // r10d
  int v59; // eax
  int v60; // esi
  unsigned int v61; // r13d
  __int64 v62; // rdi
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  unsigned int *v65; // r15
  _QWORD *v66; // r12
  _QWORD *v67; // rbx
  char v68; // r14
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int16 *v72; // [rsp+8h] [rbp-98h]
  _BYTE *v73; // [rsp+20h] [rbp-80h]
  _BYTE *v74; // [rsp+28h] [rbp-78h]
  _QWORD *v75; // [rsp+30h] [rbp-70h]
  char v76; // [rsp+38h] [rbp-68h]
  int v77; // [rsp+38h] [rbp-68h]
  unsigned int v78; // [rsp+38h] [rbp-68h]
  unsigned int v79; // [rsp+38h] [rbp-68h]
  __int64 v80; // [rsp+38h] [rbp-68h]
  _QWORD *v82; // [rsp+40h] [rbp-60h]
  __int64 v83; // [rsp+48h] [rbp-58h]
  __int64 v84; // [rsp+48h] [rbp-58h]
  _QWORD *v85; // [rsp+48h] [rbp-58h]
  __int64 v86; // [rsp+50h] [rbp-50h]
  int v88; // [rsp+58h] [rbp-48h]
  unsigned int v89; // [rsp+68h] [rbp-38h] BYREF
  unsigned int v90[13]; // [rsp+6Ch] [rbp-34h] BYREF

  v6 = *(_QWORD *)(a1 + 96);
  v89 = 0;
  v7 = (__int16 *)(*(_QWORD *)(v6 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v6 + 8) + 24LL * a2 + 4));
  v8 = v7 + 1;
  v9 = *v7 + a2;
  if ( !*v7 )
    return 0;
  v11 = a1;
  v12 = (unsigned __int16)v9;
  v13 = 0;
  v86 = 0;
  v83 = a1 + 176;
  while ( 1 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v11 + 104) + 8LL * (unsigned __int16)v12);
    if ( !v14 )
      goto LABEL_10;
    v15 = *(_DWORD *)(v11 + 200);
    if ( v15 )
    {
      a6 = *(_QWORD *)(v11 + 184);
      a5 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v16 = a6 + 16 * a5;
      v17 = *(_QWORD *)v16;
      if ( v14 == *(_QWORD *)v16 )
      {
LABEL_8:
        v18 = *(_DWORD *)(v16 + 8);
        if ( v13 < v18 )
        {
          v89 = v12;
          v13 = v18;
          v86 = v14;
        }
        goto LABEL_10;
      }
      v77 = 1;
      v51 = 0;
      while ( v17 != -4096 )
      {
        if ( !v51 && v17 == -8192 )
          v51 = v16;
        a5 = (v15 - 1) & (v77 + (_DWORD)a5);
        v16 = a6 + 16LL * (unsigned int)a5;
        v17 = *(_QWORD *)v16;
        if ( v14 == *(_QWORD *)v16 )
        {
          v12 = (unsigned __int16)v12;
          goto LABEL_8;
        }
        ++v77;
      }
      if ( !v51 )
        v51 = v16;
      v52 = *(_DWORD *)(v11 + 192);
      ++*(_QWORD *)(v11 + 176);
      v53 = v52 + 1;
      if ( 4 * (v52 + 1) < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(v11 + 196) - v53 <= v15 >> 3 )
        {
          v79 = v13;
          sub_2E261E0(v83, v15);
          v59 = *(_DWORD *)(v11 + 200);
          if ( !v59 )
          {
LABEL_121:
            ++*(_DWORD *)(v11 + 192);
            BUG();
          }
          v60 = v59 - 1;
          a5 = 0;
          v13 = v79;
          a6 = 1;
          v61 = (v59 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v62 = *(_QWORD *)(v11 + 184);
          v53 = *(_DWORD *)(v11 + 192) + 1;
          v51 = v62 + 16LL * v61;
          v63 = *(_QWORD *)v51;
          if ( v14 != *(_QWORD *)v51 )
          {
            while ( v63 != -4096 )
            {
              if ( !a5 && v63 == -8192 )
                a5 = v51;
              v61 = v60 & (a6 + v61);
              v51 = v62 + 16LL * v61;
              v63 = *(_QWORD *)v51;
              if ( v14 == *(_QWORD *)v51 )
                goto LABEL_68;
              a6 = (unsigned int)(a6 + 1);
            }
            if ( a5 )
              v51 = a5;
          }
        }
        goto LABEL_68;
      }
    }
    else
    {
      ++*(_QWORD *)(v11 + 176);
    }
    v78 = v13;
    sub_2E261E0(v83, 2 * v15);
    v54 = *(_DWORD *)(v11 + 200);
    if ( !v54 )
      goto LABEL_121;
    v55 = v54 - 1;
    a5 = *(_QWORD *)(v11 + 184);
    v13 = v78;
    v56 = (v54 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v53 = *(_DWORD *)(v11 + 192) + 1;
    v51 = a5 + 16LL * v56;
    v57 = *(_QWORD *)v51;
    if ( v14 != *(_QWORD *)v51 )
    {
      v58 = 1;
      a6 = 0;
      while ( v57 != -4096 )
      {
        if ( !a6 && v57 == -8192 )
          a6 = v51;
        v56 = v55 & (v58 + v56);
        v51 = a5 + 16LL * v56;
        v57 = *(_QWORD *)v51;
        if ( v14 == *(_QWORD *)v51 )
          goto LABEL_68;
        ++v58;
      }
      if ( a6 )
        v51 = a6;
    }
LABEL_68:
    *(_DWORD *)(v11 + 192) = v53;
    if ( *(_QWORD *)v51 != -4096 )
      --*(_DWORD *)(v11 + 196);
    *(_QWORD *)v51 = v14;
    *(_DWORD *)(v51 + 8) = 0;
LABEL_10:
    v19 = *v8++;
    if ( !(_WORD)v19 )
      break;
    v9 += v19;
    v12 = (unsigned __int16)v9;
  }
  if ( v86 )
  {
    if ( *(_QWORD *)(a3 + 72) )
    {
      sub_2DCBF00(a3 + 32, &v89);
    }
    else
    {
      v20 = *(unsigned int *)(a3 + 8);
      v21 = (unsigned int *)(*(_QWORD *)a3 + 4 * v20);
      if ( *(unsigned int **)a3 == v21 )
      {
        if ( v20 <= 3 )
        {
          v22 = v89;
          goto LABEL_87;
        }
        v66 = (_QWORD *)(a3 + 32);
      }
      else
      {
        v22 = v89;
        v23 = *(unsigned int **)a3;
        while ( *v23 != v89 )
        {
          if ( v21 == ++v23 )
            goto LABEL_86;
        }
        if ( v21 != v23 )
          goto LABEL_18;
LABEL_86:
        if ( v20 <= 3 )
        {
LABEL_87:
          v64 = v20 + 1;
          if ( v64 > *(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v64, 4u, a5, a6);
            v21 = (unsigned int *)(*(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8));
          }
          *v21 = v22;
          ++*(_DWORD *)(a3 + 8);
          goto LABEL_18;
        }
        v80 = v11;
        v65 = *(unsigned int **)a3;
        v66 = (_QWORD *)(a3 + 32);
        v67 = (_QWORD *)(a3 + 40);
        do
        {
          v70 = sub_2DCC990(v66, (__int64)v67, v65);
          if ( v71 )
          {
            v68 = v70 || (_QWORD *)v71 == v67 || *v65 < *(_DWORD *)(v71 + 32);
            v85 = (_QWORD *)v71;
            v69 = sub_22077B0(0x28u);
            *(_DWORD *)(v69 + 32) = *v65;
            sub_220F040(v68, v69, v85, v67);
            ++*(_QWORD *)(a3 + 72);
          }
          ++v65;
        }
        while ( v21 != v65 );
        v11 = v80;
      }
      *(_DWORD *)(a3 + 8) = 0;
      sub_2DCBF00((__int64)v66, &v89);
    }
LABEL_18:
    v24 = *(_BYTE **)(v86 + 32);
    v25 = &v24[40 * (*(_DWORD *)(v86 + 40) & 0xFFFFFF)];
    if ( v24 != v25 )
    {
      while ( 1 )
      {
        v26 = v24;
        if ( sub_2DADC00(v24) )
          break;
        v24 += 40;
        if ( v25 == v24 )
          return v86;
      }
      if ( v25 != v24 )
      {
        v27 = a3;
        v28 = a2;
        v84 = v11;
        v75 = (_QWORD *)(a3 + 32);
        while ( 1 )
        {
          v29 = *((unsigned int *)v26 + 2);
          if ( (_DWORD)v29 )
          {
            v30 = *(_QWORD *)(v84 + 96);
            v31 = *(_QWORD *)(v30 + 56);
            v32 = *(_QWORD *)(v30 + 8) + 24LL * (unsigned int)v29;
            v33 = (__int16 *)(v31 + 2LL * *(unsigned int *)(v32 + 8));
            v34 = v33 + 1;
            v35 = *v33;
            v36 = v29 + *v33;
            if ( v35 )
            {
              if ( v28 == v36 )
              {
LABEL_36:
                if ( !(v31 + 2LL * *(unsigned int *)(v32 + 4)) )
                  goto LABEL_28;
                v38 = (unsigned __int16)v29;
                v39 = *(_QWORD *)(v27 + 72) == 0;
                v74 = v25;
                v40 = (__int16 *)(v31 + 2LL * *(unsigned int *)(v32 + 4));
                v88 = *((_DWORD *)v26 + 2);
                v73 = v26;
                v90[0] = (unsigned __int16)v29;
                if ( !v39 )
                  goto LABEL_45;
LABEL_38:
                v41 = *(unsigned int *)(v27 + 8);
                v42 = (unsigned int *)(*(_QWORD *)v27 + 4 * v41);
                if ( *(unsigned int **)v27 == v42 )
                {
                  if ( v41 <= 3 )
                    goto LABEL_49;
                }
                else
                {
                  v43 = *(unsigned int **)v27;
                  while ( v38 != *v43 )
                  {
                    if ( v42 == ++v43 )
                      goto LABEL_48;
                  }
                  if ( v43 != v42 )
                    goto LABEL_43;
LABEL_48:
                  if ( v41 <= 3 )
                  {
LABEL_49:
                    v45 = v41 + 1;
                    if ( v45 > *(unsigned int *)(v27 + 12) )
                    {
                      sub_C8D5F0(v27, (const void *)(v27 + 16), v45, 4u, v29, v32);
                      v42 = (unsigned int *)(*(_QWORD *)v27 + 4LL * *(unsigned int *)(v27 + 8));
                    }
                    *v42 = v38;
                    ++*(_DWORD *)(v27 + 8);
LABEL_43:
                    v44 = *v40++;
                    if ( (_WORD)v44 )
                      goto LABEL_44;
LABEL_47:
                    v25 = v74;
                    v26 = v73;
                    goto LABEL_28;
                  }
                  v72 = v40;
                  v46 = *(unsigned int **)v27;
                  do
                  {
                    v49 = sub_2DCC990(v75, v27 + 40, v46);
                    if ( v50 )
                    {
                      v47 = v49 || v50 == v27 + 40 || *v46 < *(_DWORD *)(v50 + 32);
                      v76 = v47;
                      v82 = (_QWORD *)v50;
                      v48 = sub_22077B0(0x28u);
                      *(_DWORD *)(v48 + 32) = *v46;
                      sub_220F040(v76, v48, v82, (_QWORD *)(v27 + 40));
                      ++*(_QWORD *)(v27 + 72);
                    }
                    ++v46;
                  }
                  while ( v42 != v46 );
                  v40 = v72;
                }
                *(_DWORD *)(v27 + 8) = 0;
                sub_2DCBE50((__int64)v75, v90);
                while ( 1 )
                {
                  v44 = *v40++;
                  if ( !(_WORD)v44 )
                    goto LABEL_47;
LABEL_44:
                  v88 += v44;
                  v39 = *(_QWORD *)(v27 + 72) == 0;
                  v38 = (unsigned __int16)v88;
                  v90[0] = (unsigned __int16)v88;
                  if ( v39 )
                    goto LABEL_38;
LABEL_45:
                  sub_2DCBE50((__int64)v75, v90);
                }
              }
              while ( 1 )
              {
                v36 += *v34;
                if ( !*v34 )
                  break;
                ++v34;
                if ( v28 == v36 )
                  goto LABEL_36;
              }
            }
          }
LABEL_28:
          v37 = v26 + 40;
          if ( v26 + 40 != v25 )
          {
            while ( 1 )
            {
              v26 = v37;
              if ( sub_2DADC00(v37) )
                break;
              v37 += 40;
              if ( v25 == v37 )
                return v86;
            }
            if ( v25 != v37 )
              continue;
          }
          return v86;
        }
      }
    }
  }
  return v86;
}
