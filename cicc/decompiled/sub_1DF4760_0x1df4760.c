// Function: sub_1DF4760
// Address: 0x1df4760
//
void __fastcall sub_1DF4760(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // r13
  __int64 i; // rbx
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // rdi
  unsigned int v11; // r8d
  unsigned int v12; // esi
  int *v13; // rdx
  int v14; // r10d
  __int64 v15; // r11
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r11
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned int v21; // ecx
  __int64 v22; // rax
  int v23; // edx
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // r8d
  int v28; // edi
  __int64 v29; // rsi
  __int64 v30; // r9
  _WORD *v31; // r10
  unsigned __int16 v32; // cx
  __int16 *v33; // rsi
  _WORD *v34; // r10
  int v35; // r8d
  unsigned __int16 *v36; // r9
  unsigned int v37; // r10d
  unsigned int j; // edi
  bool v39; // cf
  __int16 *v40; // r10
  __int16 v41; // si
  int v42; // edi
  __int64 v43; // rax
  __int64 v44; // r12
  unsigned int v45; // ebx
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // r13
  __int64 v49; // r12
  __int64 *v50; // rax
  __int64 **v51; // rdx
  __int64 v52; // rax
  int v53; // eax
  int v54; // edx
  int v55; // r11d
  __int64 v56; // [rsp+8h] [rbp-58h]
  unsigned int v57; // [rsp+10h] [rbp-50h]
  __int64 v58; // [rsp+10h] [rbp-50h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+18h] [rbp-48h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  __int64 v62; // [rsp+18h] [rbp-48h]
  unsigned int v63; // [rsp+20h] [rbp-40h]
  __int64 v64; // [rsp+20h] [rbp-40h]
  __int64 v65; // [rsp+28h] [rbp-38h]
  __int64 v66; // [rsp+28h] [rbp-38h]

  v2 = *(unsigned int *)(a2 + 40);
  if ( (_DWORD)v2 )
  {
    v4 = a2;
    for ( i = 0; v2 != i; ++i )
    {
      v6 = *(_QWORD *)(v4 + 32);
      v7 = v6 + 40 * i;
      if ( !*(_BYTE *)v7
        && (*(_WORD *)(v7 + 2) & 0xFF0) == 0
        && (*(_BYTE *)(v7 + 4) & 1) == 0
        && (*(_BYTE *)(v7 + 3) & 0x30) == 0 )
      {
        if ( *(_DWORD *)(v7 + 8) )
        {
          if ( (unsigned __int8)sub_1E31310(v6 + 40 * i) )
          {
            v8 = *(unsigned int *)(a1 + 440);
            if ( (_DWORD)v8 )
            {
              v9 = *(_DWORD *)(v7 + 8);
              v10 = *(_QWORD *)(a1 + 424);
              v11 = i;
              v12 = (v8 - 1) & (37 * v9);
              v13 = (int *)(v10 + 16LL * v12);
              v14 = *v13;
              if ( v9 != *v13 )
              {
                v54 = 1;
                while ( v14 != -1 )
                {
                  v55 = v54 + 1;
                  v12 = (v8 - 1) & (v54 + v12);
                  v13 = (int *)(v10 + 16LL * v12);
                  v14 = *v13;
                  if ( v9 == *v13 )
                    goto LABEL_12;
                  v54 = v55;
                }
                continue;
              }
LABEL_12:
              if ( v13 != (int *)(v10 + 16 * v8) )
              {
                v15 = *(_QWORD *)(*((_QWORD *)v13 + 1) + 32LL);
                v65 = *((_QWORD *)v13 + 1);
                if ( v9 == *(_DWORD *)(v15 + 8) )
                {
                  v16 = *(_QWORD *)(a1 + 248);
                  v63 = *(_DWORD *)(v15 + 48);
                  if ( (*(_QWORD *)(*(_QWORD *)(v16 + 304) + 8LL * (v63 >> 6)) & (1LL << v63)) != 0 )
                  {
                    v59 = *(_QWORD *)(*((_QWORD *)v13 + 1) + 32LL);
                    if ( !(unsigned __int8)sub_1E69FD0(v16) )
                      continue;
                    v15 = v59;
                    v11 = i;
                    v57 = *(_DWORD *)(*(_QWORD *)(v65 + 32) + 48LL);
                  }
                  else
                  {
                    v57 = *(_DWORD *)(v15 + 48);
                  }
                  v60 = v15;
                  v17 = sub_1E16DA0(v4, v11, *(_QWORD *)(a1 + 240), *(_QWORD *)(a1 + 232));
                  v18 = v60;
                  if ( v17 )
                  {
                    v19 = *(_QWORD *)v17;
                    v20 = v57;
                    v21 = *(unsigned __int16 *)(*(_QWORD *)v17 + 22LL);
                    v22 = v57 >> 3;
                    if ( (unsigned int)v22 < v21 )
                    {
                      v23 = *(unsigned __int8 *)(*(_QWORD *)(v19 + 8) + v22);
                      if ( _bittest(&v23, v57 & 7) )
                        goto LABEL_22;
                    }
                  }
                  else if ( **(_WORD **)(v4 + 16) == 15 )
                  {
                    v50 = (__int64 *)sub_1F4ABE0(
                                       *(_QWORD *)(a1 + 232),
                                       *(unsigned int *)(*(_QWORD *)(v4 + 32) + 8LL),
                                       1);
                    v18 = v60;
                    v51 = (__int64 **)v50[4];
                    v20 = v57 >> 3;
                    while ( 1 )
                    {
                      v52 = *v50;
                      if ( v57 >> 3 < *(unsigned __int16 *)(v52 + 22) )
                      {
                        v53 = *(unsigned __int8 *)(*(_QWORD *)(v52 + 8) + v20);
                        if ( _bittest(&v53, v57 & 7) )
                          break;
                      }
                      v50 = *v51++;
                      if ( !v50 )
                        goto LABEL_3;
                    }
LABEL_22:
                    v56 = v18;
                    v58 = *(_QWORD *)(v4 + 32);
                    v61 = v58 + 40LL * *(unsigned int *)(v4 + 40);
                    v24 = sub_1E163A0(v4, v20, v61);
                    v25 = v61;
                    v26 = v58 + 40LL * v24;
                    if ( v61 == v26 )
                    {
LABEL_40:
                      sub_1E310D0(v7, v63);
                      if ( !(unsigned __int8)sub_1E31310(v56 + 40) )
                        sub_1E31360(v7, 0);
                      v43 = *(_QWORD *)(v4 + 8);
                      v44 = v65;
                      if ( v65 != v43 )
                      {
                        v66 = i;
                        v45 = v63;
                        v64 = v2;
                        v46 = a1;
                        v47 = v4;
                        v48 = v44;
                        v49 = v43;
                        do
                        {
                          sub_1E1A450(v48, v45, *(_QWORD *)(v46 + 232));
                          v48 = *(_QWORD *)(v48 + 8);
                        }
                        while ( v48 != v49 );
                        v4 = v47;
                        i = v66;
                        a1 = v46;
                        v2 = v64;
                      }
                      *(_BYTE *)(a1 + 512) = 1;
                    }
                    else
                    {
                      while ( 1 )
                      {
                        if ( v7 != v26
                          && !*(_BYTE *)v26
                          && (*(_BYTE *)(v26 + 3) & 0x20) != 0
                          && (*(_BYTE *)(v26 + 3) & 0x10) == 0 )
                        {
                          v27 = *(_DWORD *)(v26 + 8);
                          v28 = *(_DWORD *)(v7 + 8);
                          if ( v27 == v28 )
                            goto LABEL_3;
                          if ( v28 >= 0 && v27 >= 0 )
                            break;
                        }
LABEL_24:
                        v26 += 40;
                        if ( v25 == v26 )
                          goto LABEL_40;
                      }
                      v29 = *(_QWORD *)(a1 + 232);
                      v30 = *(_QWORD *)(v29 + 8);
                      v62 = *(_QWORD *)(v29 + 56);
                      LODWORD(v29) = *(_DWORD *)(v30 + 24LL * (unsigned int)v28 + 16);
                      v31 = (_WORD *)(v62 + 2LL * ((unsigned int)v29 >> 4));
                      v32 = *v31 + v28 * (v29 & 0xF);
                      v33 = v31 + 1;
                      LODWORD(v30) = *(_DWORD *)(v30 + 24LL * (unsigned int)v27 + 16);
                      v35 = (v30 & 0xF) * v27;
                      v34 = (_WORD *)(v62 + 2LL * ((unsigned int)v30 >> 4));
                      LOWORD(v35) = *v34 + v35;
                      v36 = v34 + 1;
                      v37 = v32;
                      for ( j = (unsigned __int16)v35; ; j = (unsigned __int16)v35 )
                      {
                        v39 = v37 < j;
                        if ( v37 == j )
                          break;
                        while ( v39 )
                        {
                          v40 = v33 + 1;
                          v41 = *v33;
                          v32 += v41;
                          if ( !v41 )
                            goto LABEL_24;
                          v33 = v40;
                          v37 = v32;
                          v39 = v32 < j;
                          if ( v32 == j )
                            goto LABEL_3;
                        }
                        v42 = *v36;
                        if ( !(_WORD)v42 )
                          goto LABEL_24;
                        v35 += v42;
                        ++v36;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
LABEL_3:
      ;
    }
  }
}
