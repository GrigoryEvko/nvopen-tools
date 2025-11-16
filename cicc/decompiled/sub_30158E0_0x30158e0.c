// Function: sub_30158E0
// Address: 0x30158e0
//
void __fastcall sub_30158E0(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // zf
  int v10; // eax
  __int64 v11; // r13
  unsigned __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r14
  unsigned __int64 v21; // rax
  int v22; // edx
  unsigned __int64 v23; // rax
  bool v24; // cf
  __int64 v25; // rdx
  __int64 v26; // r11
  unsigned __int64 v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // r9
  unsigned int v32; // r8d
  _QWORD *v33; // rax
  __int64 v34; // rdi
  _DWORD *v35; // rax
  char v36; // al
  unsigned __int64 v37; // rax
  __int64 v38; // r15
  int v39; // eax
  unsigned int v40; // r13d
  int v41; // r14d
  int v42; // ebx
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // rdx
  unsigned __int64 v47; // r9
  __int64 v48; // rax
  int v49; // eax
  unsigned __int8 *v50; // rdi
  const char *v51; // rax
  unsigned __int64 v52; // rdx
  int v53; // edx
  int v54; // r9d
  __int64 *v55; // rcx
  int v56; // eax
  int v57; // eax
  int v58; // r9d
  int v59; // r9d
  __int64 v60; // rdi
  unsigned int v61; // edx
  int v62; // r8d
  __int64 *v63; // r10
  int v64; // r9d
  int v65; // r9d
  __int64 v66; // rdi
  int v67; // r8d
  unsigned int v68; // edx
  __int64 v69; // [rsp+0h] [rbp-C0h]
  __int64 v71; // [rsp+20h] [rbp-A0h]
  unsigned int v72; // [rsp+20h] [rbp-A0h]
  __int64 v73; // [rsp+28h] [rbp-98h]
  __int64 v74; // [rsp+28h] [rbp-98h]
  int v75; // [rsp+28h] [rbp-98h]
  __int64 v76; // [rsp+28h] [rbp-98h]
  __int64 v77; // [rsp+28h] [rbp-98h]
  __int64 v78; // [rsp+38h] [rbp-88h] BYREF
  _BYTE *v79; // [rsp+40h] [rbp-80h] BYREF
  __int64 v80; // [rsp+48h] [rbp-78h]
  _BYTE v81[112]; // [rsp+50h] [rbp-70h] BYREF

  v79 = v81;
  v80 = 0x800000000LL;
  v3 = sub_22077B0(0x10u);
  v6 = v3;
  if ( v3 )
  {
    *(_QWORD *)v3 = a1;
    *(_DWORD *)(v3 + 8) = a2;
  }
  v7 = (unsigned int)v80;
  v8 = (unsigned int)v80 + 1LL;
  if ( v8 > HIDWORD(v80) )
  {
    sub_C8D5F0((__int64)&v79, v81, v8, 8u, v4, v5);
    v7 = (unsigned int)v80;
  }
  *(_QWORD *)&v79[8 * v7] = v6;
  v9 = (_DWORD)v80 == -1;
  v10 = v80 + 1;
  LODWORD(v80) = v80 + 1;
  if ( !v9 )
  {
    v11 = a3;
    v69 = a3 + 128;
    while ( 1 )
    {
      v12 = *(_QWORD *)&v79[8 * v10 - 8];
      LODWORD(v80) = v10 - 1;
      v13 = *(_QWORD *)v12;
      v14 = *(int *)(v12 + 8);
      j_j___libc_free_0(v12);
      v15 = *(unsigned int *)(v11 + 152);
      v16 = *(_QWORD *)(v11 + 136);
      if ( !(_DWORD)v15 )
        goto LABEL_12;
      v17 = (v15 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v13 == *v18 )
      {
LABEL_10:
        if ( v18 == (__int64 *)(v16 + 16 * v15) || *((_DWORD *)v18 + 2) > (int)v14 )
          goto LABEL_12;
LABEL_7:
        v10 = v80;
        if ( !(_DWORD)v80 )
          break;
      }
      else
      {
        v53 = 1;
        while ( v19 != -4096 )
        {
          v54 = v53 + 1;
          v17 = (v15 - 1) & (v53 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v13 == *v18 )
            goto LABEL_10;
          v53 = v54;
        }
LABEL_12:
        v20 = sub_AA4FF0(v13);
        v21 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v13 + 48 == v21 )
        {
          v26 = 0;
        }
        else
        {
          if ( !v21 )
            goto LABEL_101;
          v22 = *(unsigned __int8 *)(v21 - 24);
          v23 = v21 - 24;
          v24 = (unsigned int)(v22 - 30) < 0xB;
          v25 = 0;
          if ( v24 )
            v25 = v23;
          v26 = v25;
        }
        if ( !v20 )
LABEL_101:
          BUG();
        v27 = (unsigned int)*(unsigned __int8 *)(v20 - 24) - 39;
        if ( (unsigned int)v27 <= 0x38 )
        {
          v28 = 0x100060000000001LL;
          if ( _bittest64(&v28, v27) )
          {
            v73 = v26;
            v78 = v20 - 24;
            v29 = sub_3014430(v11, &v78);
            v26 = v73;
            v14 = *(int *)v29;
          }
        }
        v30 = *(unsigned int *)(v11 + 152);
        if ( (_DWORD)v30 )
        {
          v31 = *(_QWORD *)(v11 + 136);
          v32 = (v30 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v33 = (_QWORD *)(v31 + 16LL * v32);
          v34 = *v33;
          if ( v13 == *v33 )
          {
LABEL_23:
            v35 = v33 + 1;
            goto LABEL_24;
          }
          v75 = 1;
          v55 = 0;
          while ( v34 != -4096 )
          {
            if ( v34 == -8192 && !v55 )
              v55 = v33;
            v32 = (v30 - 1) & (v75 + v32);
            v33 = (_QWORD *)(v31 + 16LL * v32);
            v34 = *v33;
            if ( v13 == *v33 )
              goto LABEL_23;
            ++v75;
          }
          if ( !v55 )
            v55 = v33;
          v56 = *(_DWORD *)(v11 + 144);
          ++*(_QWORD *)(v11 + 128);
          v57 = v56 + 1;
          if ( 4 * v57 < (unsigned int)(3 * v30) )
          {
            if ( (int)v30 - *(_DWORD *)(v11 + 148) - v57 > (unsigned int)v30 >> 3 )
              goto LABEL_70;
            v72 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
            v77 = v26;
            sub_FF1B10(v69, v30);
            v64 = *(_DWORD *)(v11 + 152);
            if ( !v64 )
            {
LABEL_100:
              ++*(_DWORD *)(a3 + 144);
              BUG();
            }
            v65 = v64 - 1;
            v63 = 0;
            v66 = *(_QWORD *)(v11 + 136);
            v26 = v77;
            v67 = 1;
            v68 = v65 & v72;
            v57 = *(_DWORD *)(v11 + 144) + 1;
            v55 = (__int64 *)(v66 + 16LL * (v65 & v72));
            v30 = *v55;
            if ( v13 == *v55 )
              goto LABEL_70;
            while ( v30 != -4096 )
            {
              if ( !v63 && v30 == -8192 )
                v63 = v55;
              v68 = v65 & (v67 + v68);
              v55 = (__int64 *)(v66 + 16LL * v68);
              v30 = *v55;
              if ( v13 == *v55 )
                goto LABEL_70;
              ++v67;
            }
            goto LABEL_78;
          }
        }
        else
        {
          ++*(_QWORD *)(v11 + 128);
        }
        v76 = v26;
        sub_FF1B10(v69, 2 * v30);
        v58 = *(_DWORD *)(v11 + 152);
        if ( !v58 )
          goto LABEL_100;
        v59 = v58 - 1;
        v60 = *(_QWORD *)(v11 + 136);
        v26 = v76;
        v61 = v59 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v57 = *(_DWORD *)(v11 + 144) + 1;
        v55 = (__int64 *)(v60 + 16LL * v61);
        v30 = *v55;
        if ( v13 == *v55 )
          goto LABEL_70;
        v62 = 1;
        v63 = 0;
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v63 )
            v63 = v55;
          v61 = v59 & (v62 + v61);
          v55 = (__int64 *)(v60 + 16LL * v61);
          v30 = *v55;
          if ( v13 == *v55 )
            goto LABEL_70;
          ++v62;
        }
LABEL_78:
        if ( v63 )
          v55 = v63;
LABEL_70:
        *(_DWORD *)(v11 + 144) = v57;
        if ( *v55 != -4096 )
          --*(_DWORD *)(v11 + 148);
        *v55 = v13;
        v35 = v55 + 1;
        *((_DWORD *)v55 + 2) = 0;
LABEL_24:
        *v35 = v14;
        v36 = *(_BYTE *)v26;
        if ( *(_BYTE *)(v20 - 24) == 81 && v36 == 38 )
        {
          v50 = sub_BD3990(*(unsigned __int8 **)(v20 - 32LL * (*(_DWORD *)(v20 - 20) & 0x7FFFFFF) - 24), v30);
          if ( !*v50 )
          {
            v51 = sub_BD5D20((__int64)v50);
            if ( v52 > 0xE
              && *(_QWORD *)v51 == 0x61636F4C73495F5FLL
              && *((_DWORD *)v51 + 2) == 2003719532
              && *((_WORD *)v51 + 6) == 28265
              && v51[14] == 100 )
            {
              goto LABEL_28;
            }
          }
LABEL_27:
          LODWORD(v14) = *(_DWORD *)(*(_QWORD *)(v11 + 512) + 24 * v14);
          goto LABEL_28;
        }
        if ( (unsigned __int8)(v36 - 37) > 1u )
        {
          if ( v36 != 34 )
            goto LABEL_28;
          v48 = *(_QWORD *)(v26 - 32);
          if ( !v48
            || *(_BYTE *)v48
            || *(_QWORD *)(v48 + 24) != *(_QWORD *)(v26 + 80)
            || (*(_BYTE *)(v48 + 33) & 0x20) == 0 )
          {
            goto LABEL_28;
          }
          v49 = *(_DWORD *)(v48 + 36);
          if ( v49 == 318 )
          {
            v78 = v26;
            LODWORD(v14) = *(_DWORD *)sub_3013480(v11 + 64, &v78);
            goto LABEL_28;
          }
          if ( v49 == 319 )
            goto LABEL_27;
        }
        else if ( (int)v14 > 0 )
        {
          goto LABEL_27;
        }
LABEL_28:
        v37 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v13 + 48 == v37 )
          goto LABEL_7;
        if ( !v37 )
          BUG();
        v38 = v37 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v37 - 24) - 30 > 0xA )
          goto LABEL_7;
        v39 = sub_B46E30(v38);
        if ( !v39 )
          goto LABEL_7;
        v74 = v11;
        v40 = 0;
        v41 = v14;
        v42 = v39;
        do
        {
          v43 = sub_B46EC0(v38, v40);
          v44 = sub_22077B0(0x10u);
          if ( v44 )
          {
            *(_QWORD *)v44 = v43;
            *(_DWORD *)(v44 + 8) = v41;
          }
          v46 = (unsigned int)v80;
          v47 = (unsigned int)v80 + 1LL;
          if ( v47 > HIDWORD(v80) )
          {
            v71 = v44;
            sub_C8D5F0((__int64)&v79, v81, (unsigned int)v80 + 1LL, 8u, v45, v47);
            v46 = (unsigned int)v80;
            v44 = v71;
          }
          ++v40;
          *(_QWORD *)&v79[8 * v46] = v44;
          v10 = v80 + 1;
          LODWORD(v80) = v80 + 1;
        }
        while ( v42 != v40 );
        v11 = v74;
        if ( !v10 )
          break;
      }
    }
  }
  if ( v79 != v81 )
    _libc_free((unsigned __int64)v79);
}
