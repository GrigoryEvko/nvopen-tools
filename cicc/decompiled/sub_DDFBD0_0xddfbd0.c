// Function: sub_DDFBD0
// Address: 0xddfbd0
//
__int64 __fastcall sub_DDFBD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // r12
  unsigned __int16 v13; // ax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  int v17; // r12d
  __int64 v18; // rax
  __int64 v19; // r15
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // r13
  __int64 *v25; // r15
  __int64 v26; // r14
  __int64 **v27; // rax
  int v28; // eax
  __int64 v29; // rsi
  unsigned int v30; // eax
  int v31; // edi
  unsigned __int16 v32; // ax
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  _BYTE *v35; // r15
  _BYTE *i; // rbx
  __int64 v37; // r12
  _BYTE *v38; // rsi
  _QWORD *v39; // rax
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 v42; // r12
  __int64 **v43; // rdx
  __int64 *v44; // r15
  __int64 **v45; // rax
  char v46; // dl
  _QWORD *v47; // rdi
  _QWORD *v48; // rcx
  _QWORD *v49; // rax
  __int64 v50; // rdx
  __int64 *v51; // rax
  char v52; // dl
  int v53; // eax
  __int64 v54; // rsi
  int v55; // edx
  unsigned int v56; // eax
  int v57; // edi
  unsigned __int16 v58; // ax
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  int v61; // [rsp+10h] [rbp-F0h]
  unsigned int v62; // [rsp+14h] [rbp-ECh]
  _BYTE *v65; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v66; // [rsp+38h] [rbp-C8h]
  _BYTE v67[48]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v68; // [rsp+70h] [rbp-90h] BYREF
  __int64 **v69; // [rsp+78h] [rbp-88h]
  __int64 v70; // [rsp+80h] [rbp-80h]
  int v71; // [rsp+88h] [rbp-78h]
  char v72; // [rsp+8Ch] [rbp-74h]
  __int64 v73; // [rsp+90h] [rbp-70h] BYREF

  v6 = a1 + 1000;
  if ( !(_DWORD)a3 )
    v6 = a1 + 968;
  v65 = v67;
  v66 = 0x600000000LL;
  v69 = (__int64 **)&v73;
  v70 = 0x100000008LL;
  v7 = *(_DWORD *)(v6 + 24);
  v62 = a3;
  v71 = 0;
  v72 = 1;
  v73 = a2;
  v68 = 1;
  if ( v7 )
  {
    a3 = (unsigned int)(v7 - 1);
    v8 = *(_QWORD *)(v6 + 8);
    a6 = a2;
    v9 = a3 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = *(_QWORD *)(v8 + 40LL * v9);
    if ( a6 == v10 )
      goto LABEL_5;
    a4 = 1;
    while ( v10 != -4096 )
    {
      a5 = (unsigned int)(a4 + 1);
      v9 = a3 & (a4 + v9);
      v10 = *(_QWORD *)(v8 + 40LL * v9);
      if ( a2 == v10 )
        goto LABEL_5;
      a4 = (unsigned int)a5;
    }
  }
  v13 = *(_WORD *)(a2 + 24);
  if ( v13 == 15 )
  {
    if ( **(_BYTE **)(a2 - 8) != 84 )
      goto LABEL_5;
    goto LABEL_14;
  }
  if ( v13 <= 0xFu )
  {
LABEL_14:
    sub_D9B3A0((__int64)&v65, a2, a3, a4, a5, a6);
    if ( !(_DWORD)v66 )
      goto LABEL_5;
    v17 = 0;
    v18 = 0;
    while ( 1 )
    {
      v20 = v65;
      v21 = *(_QWORD *)&v65[8 * v18];
      if ( *(_WORD *)(v21 + 24) == 15 )
      {
        v19 = *(_QWORD *)(v21 - 8);
        if ( *(_BYTE *)v19 == 84 )
        {
          if ( !*(_BYTE *)(a1 + 444) )
            goto LABEL_68;
          v39 = *(_QWORD **)(a1 + 424);
          v14 = *(unsigned int *)(a1 + 436);
          v20 = &v39[v14];
          if ( v39 == v20 )
          {
LABEL_55:
            if ( (unsigned int)v14 >= *(_DWORD *)(a1 + 432) )
            {
LABEL_68:
              sub_C8CC70(a1 + 416, *(_QWORD *)(v21 - 8), (__int64)v20, v14, v15, v16);
              if ( !v46 )
                goto LABEL_17;
            }
            else
            {
              v14 = (unsigned int)(v14 + 1);
              *(_DWORD *)(a1 + 436) = v14;
              *v20 = v19;
              ++*(_QWORD *)(a1 + 416);
            }
            v40 = 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(v19 + 7) & 0x40) != 0 )
            {
              v41 = *(_QWORD *)(v19 - 8);
              v19 = v41 + v40;
            }
            else
            {
              v41 = v19 - v40;
            }
            if ( v19 == v41 )
              goto LABEL_17;
            v61 = v17;
            v42 = v19;
            while ( 2 )
            {
              v44 = sub_DD8400(a1, *(_QWORD *)(v42 - 32));
              if ( !v72 )
                goto LABEL_83;
              v45 = v69;
              v14 = HIDWORD(v70);
              v43 = &v69[HIDWORD(v70)];
              if ( v69 != v43 )
              {
                while ( v44 != *v45 )
                {
                  if ( v43 == ++v45 )
                    goto LABEL_93;
                }
                goto LABEL_66;
              }
LABEL_93:
              if ( HIDWORD(v70) < (unsigned int)v70 )
              {
                v14 = (unsigned int)++HIDWORD(v70);
                *v43 = v44;
                ++v68;
LABEL_84:
                v53 = *(_DWORD *)(v6 + 24);
                v54 = *(_QWORD *)(v6 + 8);
                if ( !v53 )
                {
LABEL_88:
                  v58 = *((_WORD *)v44 + 12);
                  if ( v58 == 15 )
                  {
                    if ( *(_BYTE *)*(v44 - 1) != 84 )
                      goto LABEL_66;
                  }
                  else if ( v58 > 0xFu )
                  {
                    if ( v58 == 16 )
                      goto LABEL_105;
                    goto LABEL_66;
                  }
                  v59 = (unsigned int)v66;
                  v14 = HIDWORD(v66);
                  v60 = (unsigned int)v66 + 1LL;
                  if ( v60 > HIDWORD(v66) )
                  {
                    sub_C8D5F0((__int64)&v65, v67, v60, 8u, v15, v16);
                    v59 = (unsigned int)v66;
                  }
                  *(_QWORD *)&v65[8 * v59] = v44;
                  LODWORD(v66) = v66 + 1;
                  goto LABEL_66;
                }
                v55 = v53 - 1;
                v56 = (v53 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                v14 = *(_QWORD *)(v54 + 40LL * v56);
                if ( v44 != (__int64 *)v14 )
                {
                  v57 = 1;
                  while ( v14 != -4096 )
                  {
                    v15 = (unsigned int)(v57 + 1);
                    v56 = v55 & (v57 + v56);
                    v14 = *(_QWORD *)(v54 + 40LL * v56);
                    if ( v44 == (__int64 *)v14 )
                      goto LABEL_66;
                    ++v57;
                  }
                  goto LABEL_88;
                }
              }
              else
              {
LABEL_83:
                sub_C8CC70((__int64)&v68, (__int64)v44, (__int64)v43, v14, v15, v16);
                if ( v52 )
                  goto LABEL_84;
              }
LABEL_66:
              v42 -= 32;
              if ( v42 == v41 )
              {
                v17 = v61;
                goto LABEL_17;
              }
              continue;
            }
          }
          while ( v19 != *v39 )
          {
            if ( v20 == ++v39 )
              goto LABEL_55;
          }
        }
      }
      else
      {
        v22 = (__int64 *)sub_D960E0(v21);
        v24 = &v22[v23];
        v25 = v22;
        if ( v24 != v22 )
        {
          v26 = *v22;
          if ( v72 )
          {
LABEL_21:
            v27 = v69;
            v14 = HIDWORD(v70);
            v23 = (__int64)&v69[HIDWORD(v70)];
            if ( v69 == (__int64 **)v23 )
              goto LABEL_38;
            while ( (__int64 *)v26 != *v27 )
            {
              if ( (__int64 **)v23 == ++v27 )
              {
LABEL_38:
                if ( HIDWORD(v70) < (unsigned int)v70 )
                {
                  v14 = (unsigned int)++HIDWORD(v70);
                  *(_QWORD *)v23 = v26;
                  ++v68;
                  goto LABEL_28;
                }
                goto LABEL_27;
              }
            }
            goto LABEL_25;
          }
LABEL_27:
          while ( 2 )
          {
            sub_C8CC70((__int64)&v68, v26, v23, v14, v15, v16);
            if ( !(_BYTE)v23 )
              goto LABEL_25;
LABEL_28:
            v28 = *(_DWORD *)(v6 + 24);
            v29 = *(_QWORD *)(v6 + 8);
            if ( !v28 )
              goto LABEL_32;
            v23 = (unsigned int)(v28 - 1);
            v30 = v23 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v14 = *(_QWORD *)(v29 + 40LL * v30);
            if ( v26 == v14 )
            {
LABEL_25:
              if ( v24 == ++v25 )
                break;
            }
            else
            {
              v31 = 1;
              while ( v14 != -4096 )
              {
                v15 = (unsigned int)(v31 + 1);
                v30 = v23 & (v31 + v30);
                v14 = *(_QWORD *)(v29 + 40LL * v30);
                if ( v26 == v14 )
                  goto LABEL_25;
                ++v31;
              }
LABEL_32:
              v32 = *(_WORD *)(v26 + 24);
              if ( v32 != 15 )
              {
                if ( v32 <= 0xFu )
                {
LABEL_34:
                  v33 = (unsigned int)v66;
                  v14 = HIDWORD(v66);
                  v34 = (unsigned int)v66 + 1LL;
                  if ( v34 > HIDWORD(v66) )
                  {
                    sub_C8D5F0((__int64)&v65, v67, v34, 8u, v15, v16);
                    v33 = (unsigned int)v66;
                  }
                  v23 = (__int64)v65;
                  ++v25;
                  *(_QWORD *)&v65[8 * v33] = v26;
                  LODWORD(v66) = v66 + 1;
                  if ( v24 == v25 )
                    break;
                  goto LABEL_26;
                }
                if ( v32 == 16 )
                  goto LABEL_105;
                goto LABEL_25;
              }
              if ( **(_BYTE **)(v26 - 8) == 84 )
                goto LABEL_34;
              if ( v24 == ++v25 )
                break;
            }
LABEL_26:
            v26 = *v25;
            if ( v72 )
              goto LABEL_21;
            continue;
          }
        }
      }
LABEL_17:
      v18 = (unsigned int)(v17 + 1);
      v17 = v18;
      if ( (_DWORD)v66 == (_DWORD)v18 )
      {
        if ( (_DWORD)v18 )
        {
          v35 = &v65[8 * v18];
          for ( i = v65 + 8; v35 != i; v35 -= 8 )
          {
            v37 = *((_QWORD *)v35 - 1);
            sub_DBB9F0(a1, v37, v62, 0);
            if ( *(_WORD *)(v37 + 24) == 15 )
            {
              v38 = *(_BYTE **)(v37 - 8);
              if ( *v38 == 84 )
              {
                if ( *(_BYTE *)(a1 + 444) )
                {
                  v47 = *(_QWORD **)(a1 + 424);
                  v48 = &v47[*(unsigned int *)(a1 + 436)];
                  if ( v47 != v48 )
                  {
                    v49 = *(_QWORD **)(a1 + 424);
                    while ( v38 != (_BYTE *)*v49 )
                    {
                      if ( v48 == ++v49 )
                        goto LABEL_49;
                    }
                    v50 = (unsigned int)(*(_DWORD *)(a1 + 436) - 1);
                    *(_DWORD *)(a1 + 436) = v50;
                    *v49 = v47[v50];
                    ++*(_QWORD *)(a1 + 416);
                  }
                }
                else
                {
                  v51 = sub_C8CA60(a1 + 416, (__int64)v38);
                  if ( v51 )
                  {
                    *v51 = -2;
                    ++*(_DWORD *)(a1 + 440);
                    ++*(_QWORD *)(a1 + 416);
                  }
                }
              }
            }
LABEL_49:
            ;
          }
        }
        goto LABEL_5;
      }
    }
  }
  if ( v13 == 16 )
LABEL_105:
    BUG();
LABEL_5:
  v11 = sub_DBB9F0(a1, a2, v62, 0);
  if ( !v72 )
    _libc_free(v69, a2);
  if ( v65 != v67 )
    _libc_free(v65, a2);
  return v11;
}
