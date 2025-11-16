// Function: sub_1E1F460
// Address: 0x1e1f460
//
__int64 __fastcall sub_1E1F460(__int64 a1, __int64 a2, __int64 a3, char a4, int a5, int a6)
{
  _WORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r14
  unsigned int v11; // r13d
  char v12; // r15
  unsigned __int64 v13; // r13
  int v14; // edx
  char v15; // al
  int v16; // r15d
  unsigned int *v18; // r13
  unsigned int v19; // eax
  unsigned int v20; // r10d
  __int64 v21; // rdi
  unsigned int v22; // ecx
  int *v23; // rdx
  int v24; // r9d
  int *v25; // r11
  int v26; // eax
  unsigned int v27; // esi
  _DWORD *v28; // rax
  int v29; // ecx
  unsigned int v30; // esi
  int v31; // edx
  unsigned __int64 v32; // rsi
  unsigned int *v33; // rcx
  unsigned int *v34; // rdx
  unsigned int *v35; // rax
  unsigned int v36; // r14d
  int v37; // r11d
  int v38; // r14d
  int *v39; // r11
  int v40; // eax
  int v41; // ecx
  char v42; // al
  __int64 v43; // rdx
  unsigned int v44; // ecx
  __int64 v45; // rax
  _BOOL4 v46; // r15d
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rsi
  __int64 v51; // rbx
  unsigned int *v52; // r14
  unsigned int v53; // ecx
  __int64 i; // r15
  unsigned int v55; // edi
  __int64 v56; // rax
  char v57; // r11
  _BOOL4 v58; // ebx
  __int64 v59; // rax
  unsigned int v60; // eax
  __int64 v61; // rax
  __int64 v62; // r8
  unsigned int v63; // edx
  __int64 v64; // rax
  char v65; // cl
  __int64 v66; // rax
  int v67; // r11d
  int *v68; // r14
  __int64 v69; // [rsp+10h] [rbp-80h]
  int v70; // [rsp+18h] [rbp-78h]
  __int64 v71; // [rsp+18h] [rbp-78h]
  int v72; // [rsp+20h] [rbp-70h]
  __int64 v73; // [rsp+20h] [rbp-70h]
  unsigned int v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+20h] [rbp-70h]
  __int64 v77; // [rsp+20h] [rbp-70h]
  __int64 v78; // [rsp+28h] [rbp-68h]
  unsigned int v79; // [rsp+30h] [rbp-60h]
  char v80; // [rsp+36h] [rbp-5Ah]
  __int64 v83; // [rsp+40h] [rbp-50h]
  unsigned int v85; // [rsp+54h] [rbp-3Ch] BYREF
  _QWORD v86[7]; // [rsp+58h] [rbp-38h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v7 = *(_WORD **)(a3 + 16);
  v80 = a5;
  if ( *v7 != 9 )
  {
    v8 = (unsigned __int16)v7[1];
    if ( (_WORD)v8 )
    {
      v9 = 0;
      v83 = 40 * v8;
      v78 = a2 + 896;
      do
      {
        v10 = v9 + *(_QWORD *)(a3 + 32);
        if ( *(_BYTE *)v10 )
          goto LABEL_11;
        if ( (*(_BYTE *)(v10 + 3) & 0x20) != 0 )
          goto LABEL_11;
        v11 = *(_DWORD *)(v10 + 8);
        if ( (v11 & 0x80000000) == 0 )
          goto LABEL_11;
        v12 = 0;
        if ( !a4 )
          goto LABEL_8;
        if ( *(_QWORD *)(a2 + 928) )
        {
          v43 = *(_QWORD *)(a2 + 904);
          if ( v43 )
          {
            while ( 1 )
            {
              v44 = *(_DWORD *)(v43 + 32);
              v45 = *(_QWORD *)(v43 + 24);
              v12 = 0;
              if ( v11 < v44 )
              {
                v45 = *(_QWORD *)(v43 + 16);
                v12 = a4;
              }
              if ( !v45 )
                break;
              v43 = v45;
            }
            if ( !v12 )
            {
              if ( v11 <= v44 )
                goto LABEL_8;
LABEL_66:
              v46 = 1;
              if ( v78 != v43 )
                v46 = v11 < *(_DWORD *)(v43 + 32);
LABEL_68:
              v73 = v43;
              v47 = sub_22077B0(40);
              v48 = v78;
              v49 = v73;
              *(_DWORD *)(v47 + 32) = v11;
              v50 = v47;
              goto LABEL_69;
            }
            if ( *(_QWORD *)(a2 + 912) == v43 )
              goto LABEL_66;
          }
          else
          {
            v43 = v78;
            if ( v78 == *(_QWORD *)(a2 + 912) )
            {
              v43 = v78;
              v46 = 1;
              goto LABEL_68;
            }
          }
          v76 = v43;
          v12 = 0;
          if ( v11 > *(_DWORD *)(sub_220EF80(v43) + 32) )
          {
            v43 = v76;
            if ( v76 )
              goto LABEL_66;
          }
        }
        else
        {
          v32 = *(unsigned int *)(a2 + 752);
          v33 = *(unsigned int **)(a2 + 744);
          v34 = &v33[v32];
          if ( v33 != v34 )
          {
            v35 = *(unsigned int **)(a2 + 744);
            while ( v11 != *v35 )
            {
              if ( v34 == ++v35 )
                goto LABEL_73;
            }
            if ( v34 != v35 )
            {
              v12 = 0;
              goto LABEL_8;
            }
          }
LABEL_73:
          if ( v32 > 0x1F )
          {
            v69 = v9 + *(_QWORD *)(a3 + 32);
            v79 = *(_DWORD *)(v10 + 8);
            v71 = v9;
            v51 = *(_QWORD *)(a2 + 904);
            while ( 1 )
            {
              v52 = &v33[v32 - 1];
              if ( v51 )
              {
                v53 = *v52;
                for ( i = v51; ; i = v56 )
                {
                  v55 = *(_DWORD *)(i + 32);
                  v56 = *(_QWORD *)(i + 24);
                  v57 = 0;
                  if ( v53 < v55 )
                  {
                    v56 = *(_QWORD *)(i + 16);
                    v57 = a4;
                  }
                  if ( !v56 )
                    break;
                }
                if ( !v57 )
                {
                  if ( v53 <= v55 )
                    goto LABEL_86;
LABEL_83:
                  v58 = 1;
                  if ( v78 != i )
                    v58 = v53 < *(_DWORD *)(i + 32);
LABEL_85:
                  v59 = sub_22077B0(40);
                  *(_DWORD *)(v59 + 32) = *v52;
                  sub_220F040(v58, v59, i, v78);
                  v51 = *(_QWORD *)(a2 + 904);
                  ++*(_QWORD *)(a2 + 928);
                  goto LABEL_86;
                }
                if ( i == *(_QWORD *)(a2 + 912) )
                  goto LABEL_83;
              }
              else
              {
                i = v78;
                if ( v78 == *(_QWORD *)(a2 + 912) )
                {
                  i = v78;
                  v58 = 1;
                  goto LABEL_85;
                }
                v53 = *v52;
              }
              v74 = v53;
              v61 = sub_220EF80(i);
              v53 = v74;
              if ( v74 > *(_DWORD *)(v61 + 32) )
                goto LABEL_83;
LABEL_86:
              v60 = *(_DWORD *)(a2 + 752) - 1;
              *(_DWORD *)(a2 + 752) = v60;
              if ( !v60 )
              {
                v62 = v51;
                v10 = v69;
                v9 = v71;
                if ( v62 )
                {
                  while ( 1 )
                  {
                    v63 = *(_DWORD *)(v62 + 32);
                    v64 = *(_QWORD *)(v62 + 24);
                    v65 = 0;
                    if ( v79 < v63 )
                    {
                      v64 = *(_QWORD *)(v62 + 16);
                      v65 = a4;
                    }
                    if ( !v64 )
                      break;
                    v62 = v64;
                  }
                  if ( v65 )
                  {
                    if ( v62 != *(_QWORD *)(a2 + 912) )
                      goto LABEL_109;
                  }
                  else if ( v79 <= v63 )
                  {
                    goto LABEL_70;
                  }
                }
                else
                {
                  v62 = v78;
                  if ( v78 == *(_QWORD *)(a2 + 912) )
                  {
                    v62 = v78;
                    v46 = 1;
                    goto LABEL_101;
                  }
LABEL_109:
                  v77 = v62;
                  if ( v79 <= *(_DWORD *)(sub_220EF80(v62) + 32) || (v62 = v77) == 0 )
                  {
LABEL_70:
                    v12 = a4;
                    goto LABEL_8;
                  }
                }
                v46 = 1;
                if ( v78 != v62 )
                  v46 = v79 < *(_DWORD *)(v62 + 32);
LABEL_101:
                v75 = v62;
                v66 = sub_22077B0(40);
                v48 = v78;
                *(_DWORD *)(v66 + 32) = v79;
                v50 = v66;
                v49 = v75;
LABEL_69:
                sub_220F040(v46, v50, v49, v48);
                ++*(_QWORD *)(a2 + 928);
                goto LABEL_70;
              }
              v33 = *(unsigned int **)(a2 + 744);
              v32 = v60;
            }
          }
          if ( *(_DWORD *)(a2 + 752) >= *(_DWORD *)(a2 + 756) )
          {
            sub_16CD150(a2 + 744, (const void *)(a2 + 760), 0, 4, (unsigned __int8)a5, a6);
            v34 = (unsigned int *)(*(_QWORD *)(a2 + 744) + 4LL * *(unsigned int *)(a2 + 752));
          }
          v12 = a4;
          *v34 = v11;
          ++*(_DWORD *)(a2 + 752);
        }
LABEL_8:
        v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 264) + 24LL) + 16LL * (v11 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
        v14 = *(_DWORD *)(*(__int64 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a2 + 248) + 184LL))(
                           *(_QWORD *)(a2 + 248),
                           v13);
        v15 = *(_BYTE *)(v10 + 3);
        if ( (v15 & 0x10) == 0 )
        {
          if ( (v15 & 0x40) == 0 )
          {
            v72 = v14;
            v42 = sub_1E69E00(*(_QWORD *)(a2 + 264));
            v14 = v72;
            if ( !v42 )
            {
              if ( !v12 )
                goto LABEL_11;
              v16 = v72;
              if ( !v80 )
                goto LABEL_11;
              goto LABEL_10;
            }
          }
          if ( v12 )
            goto LABEL_11;
          v14 = -v14;
        }
        v16 = v14;
LABEL_10:
        if ( v16 )
        {
          v18 = (unsigned int *)(*(__int64 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a2 + 248) + 224LL))(
                                  *(_QWORD *)(a2 + 248),
                                  v13);
          v19 = *v18;
          if ( *v18 != -1 )
          {
            while ( 1 )
            {
              while ( 1 )
              {
                v27 = *(_DWORD *)(a1 + 24);
                a5 = v19;
                if ( !v27 )
                  goto LABEL_21;
                v20 = v27 - 1;
                v21 = *(_QWORD *)(a1 + 8);
                v22 = (v27 - 1) & (37 * v19);
                v23 = (int *)(v21 + 8LL * v22);
                v24 = *v23;
                v25 = v23;
                if ( v19 == *v23 )
                  break;
                v36 = (v27 - 1) & (37 * v19);
                v37 = 1;
                while ( v24 != -1 )
                {
                  v36 = v20 & (v37 + v36);
                  v70 = v37 + 1;
                  v25 = (int *)(v21 + 8LL * v36);
                  v24 = *v25;
                  if ( v19 == *v25 )
                    goto LABEL_16;
                  v37 = v70;
                }
LABEL_21:
                v85 = v19;
                LOBYTE(a5) = sub_1BFD720(a1, (int *)&v85, v86);
                v28 = (_DWORD *)v86[0];
                if ( !(_BYTE)a5 )
                {
                  v29 = *(_DWORD *)(a1 + 16);
                  v30 = *(_DWORD *)(a1 + 24);
                  ++*(_QWORD *)a1;
                  v31 = v29 + 1;
                  LOBYTE(a5) = 2 * v30;
                  if ( 4 * (v29 + 1) >= 3 * v30 )
                  {
                    v30 *= 2;
                  }
                  else if ( v30 - *(_DWORD *)(a1 + 20) - v31 > v30 >> 3 )
                  {
LABEL_24:
                    *(_DWORD *)(a1 + 16) = v31;
                    if ( *v28 != -1 )
                      --*(_DWORD *)(a1 + 20);
                    *(_QWORD *)v28 = v85;
                    goto LABEL_27;
                  }
                  sub_1BFDD60(a1, v30);
                  sub_1BFD720(a1, (int *)&v85, v86);
                  v28 = (_DWORD *)v86[0];
                  v31 = *(_DWORD *)(a1 + 16) + 1;
                  goto LABEL_24;
                }
LABEL_27:
                ++v18;
                v28[1] = v16;
                v19 = *v18;
                if ( *v18 == -1 )
                  goto LABEL_11;
              }
LABEL_16:
              if ( v25 == (int *)(v21 + 8LL * v27) )
                goto LABEL_21;
              v85 = v19;
              a6 = *v23;
              if ( v19 != *v23 )
                break;
              v26 = v16 + v23[1];
LABEL_19:
              ++v18;
              v23[1] = v26;
              v19 = *v18;
              if ( *v18 == -1 )
                goto LABEL_11;
            }
            v38 = 1;
            v39 = 0;
            while ( a6 != -1 )
            {
              if ( a6 != -2 || v39 )
                v23 = v39;
              v67 = v38 + 1;
              v22 = v20 & (v38 + v22);
              v68 = (int *)(v21 + 8LL * v22);
              a6 = *v68;
              if ( v19 == *v68 )
              {
                v23 = (int *)(v21 + 8LL * v22);
                v26 = v16 + v68[1];
                goto LABEL_19;
              }
              v38 = v67;
              v39 = v23;
              v23 = (int *)(v21 + 8LL * v22);
            }
            v40 = *(_DWORD *)(a1 + 16);
            a6 = 2 * v27;
            if ( v39 )
              v23 = v39;
            ++*(_QWORD *)a1;
            v41 = v40 + 1;
            if ( 4 * (v40 + 1) >= 3 * v27 )
            {
              v27 *= 2;
            }
            else if ( v27 - *(_DWORD *)(a1 + 20) - v41 > v27 >> 3 )
            {
LABEL_49:
              *(_DWORD *)(a1 + 16) = v41;
              if ( *v23 != -1 )
                --*(_DWORD *)(a1 + 20);
              *v23 = a5;
              v26 = v16;
              v23[1] = 0;
              goto LABEL_19;
            }
            sub_1BFDD60(a1, v27);
            sub_1BFD720(a1, (int *)&v85, v86);
            v23 = (int *)v86[0];
            a5 = v85;
            v41 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_49;
          }
        }
LABEL_11:
        v9 += 40;
      }
      while ( v9 != v83 );
    }
  }
  return a1;
}
