// Function: sub_3853BF0
// Address: 0x3853bf0
//
__int64 __fastcall sub_3853BF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, int a6)
{
  __int64 v6; // r15
  __int64 v8; // rax
  __int64 v9; // r10
  __int64 v10; // rbx
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // r11
  unsigned int v17; // r14d
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rbx
  unsigned __int64 v22; // rax
  __int64 v23; // rbx
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rdi
  void *v28; // rax
  int v29; // edx
  unsigned int v30; // eax
  __int64 *v31; // rax
  _QWORD *v32; // r12
  __int64 *v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rbx
  int v37; // r12d
  _QWORD *v38; // rdi
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rax
  int v43; // eax
  int v44; // ecx
  __int64 v45; // rsi
  unsigned int v46; // eax
  __int64 v47; // rdx
  int v48; // edi
  int v49; // eax
  int v50; // edi
  __int64 *v51; // rax
  int v52; // eax
  int v53; // r9d
  int v54; // eax
  int v55; // r9d
  __int64 v56; // [rsp+8h] [rbp-98h]
  __int64 v57; // [rsp+10h] [rbp-90h]
  __int64 v58; // [rsp+18h] [rbp-88h]
  unsigned __int64 v59; // [rsp+18h] [rbp-88h]
  __int64 v60; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v61; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v62; // [rsp+38h] [rbp-68h]
  unsigned __int64 v63; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v64; // [rsp+48h] [rbp-58h]
  __int64 *v65; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v66; // [rsp+58h] [rbp-48h] BYREF
  __int64 v67[8]; // [rsp+60h] [rbp-40h] BYREF

  v6 = a2;
  v8 = *(_QWORD *)(a2 - 48);
  v65 = v67;
  v57 = v8;
  v56 = *(_QWORD *)(a2 - 24);
  v66 = 0x200000000LL;
  v9 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v10 = *(_QWORD *)(a2 - 8);
    v11 = v10 + v9;
  }
  else
  {
    v10 = a2 - v9;
    v11 = a2;
  }
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 16LL) > 0x10u )
      {
        v14 = *(unsigned int *)(a1 + 160);
        if ( !(_DWORD)v14 )
          goto LABEL_12;
        v14 = (unsigned int)(v14 - 1);
        a2 = *(_QWORD *)(a1 + 144);
        a4 = (unsigned int)v14 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v15 = (__int64 *)(a2 + 16 * a4);
        v16 = *v15;
        if ( v13 != *v15 )
        {
          v49 = 1;
          while ( v16 != -8 )
          {
            v50 = v49 + 1;
            a4 = (unsigned int)v14 & (v49 + (_DWORD)a4);
            v15 = (__int64 *)(a2 + 16LL * (unsigned int)a4);
            v16 = *v15;
            if ( v13 == *v15 )
              goto LABEL_11;
            v49 = v50;
          }
          goto LABEL_12;
        }
LABEL_11:
        v13 = v15[1];
        if ( !v13 )
          goto LABEL_12;
      }
      v12 = (unsigned int)v66;
      if ( (unsigned int)v66 >= HIDWORD(v66) )
      {
        a2 = (__int64)v67;
        v58 = v11;
        sub_16CD150((__int64)&v65, v67, 0, 8, a5, a6);
        v12 = (unsigned int)v66;
        v11 = v58;
      }
      v10 += 24;
      v65[v12] = v13;
      LODWORD(v66) = v66 + 1;
      if ( v11 == v10 )
      {
        v51 = v65;
        goto LABEL_20;
      }
    }
  }
  v51 = v67;
LABEL_20:
  a2 = *v51;
  v23 = sub_15A37B0(*(_WORD *)(v6 + 18) & 0x7FFF, (_QWORD *)*v51, (_QWORD *)v51[1], 0);
  if ( v23 )
  {
    v63 = v6;
    v17 = 1;
    sub_38526A0(a1 + 136, (__int64 *)&v63)[1] = v23;
    if ( v65 != v67 )
      _libc_free((unsigned __int64)v65);
    return v17;
  }
LABEL_12:
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  v17 = 0;
  if ( *(_BYTE *)(v6 + 16) != 76 )
  {
    v18 = *(_DWORD *)(a1 + 256);
    v62 = 1;
    v61 = 0;
    v64 = 1;
    v63 = 0;
    if ( v18 )
    {
      a2 = (unsigned int)(v18 - 1);
      a4 = *(_QWORD *)(a1 + 240);
      v14 = (unsigned int)a2 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v19 = a4 + 32 * v14;
      v20 = *(_QWORD *)v19;
      if ( v57 == *(_QWORD *)v19 )
      {
LABEL_17:
        v21 = *(_QWORD *)(v19 + 8);
        a2 = v19 + 16;
        v65 = (__int64 *)v21;
        v14 = *(unsigned int *)(v19 + 24);
        LODWORD(v67[0]) = v14;
        if ( (unsigned int)v14 > 0x40 )
        {
          sub_16A4FD0((__int64)&v66, (const void **)a2);
          v21 = (__int64)v65;
          v22 = v66;
          v14 = LODWORD(v67[0]);
        }
        else
        {
          v22 = *(_QWORD *)(v19 + 16);
        }
        v61 = v22;
        v62 = v14;
        if ( !v21 )
          goto LABEL_33;
        v25 = *(_DWORD *)(a1 + 256);
        if ( v25 )
        {
          a5 = v56;
          a2 = (unsigned int)(v25 - 1);
          a4 = *(_QWORD *)(a1 + 240);
          v14 = (unsigned int)a2 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
          v26 = a4 + 32 * v14;
          v27 = *(_QWORD *)v26;
          if ( *(_QWORD *)v26 == v56 )
          {
LABEL_29:
            v14 = *(_QWORD *)(v26 + 8);
            v65 = (__int64 *)v14;
            a4 = *(unsigned int *)(v26 + 24);
            LODWORD(v67[0]) = a4;
            if ( (unsigned int)a4 > 0x40 )
            {
              a2 = v26 + 16;
              v28 = sub_16A4FD0((__int64)&v66, (const void **)(v26 + 16));
              v14 = (__int64)v65;
            }
            else
            {
              v28 = *(void **)(v26 + 16);
              v66 = (unsigned __int64)v28;
            }
            LOBYTE(v17) = v14 != 0;
            LOBYTE(v28) = v21 == v14;
            v17 &= (unsigned int)v28;
            v63 = v66;
            v64 = v67[0];
            if ( (_BYTE)v17 )
            {
              v31 = (__int64 *)sub_16498A0(v57);
              v32 = (_QWORD *)sub_159C0E0(v31, (__int64)&v61);
              v33 = (__int64 *)sub_16498A0(v56);
              v34 = (_QWORD *)sub_159C0E0(v33, (__int64)&v63);
              a2 = (__int64)v32;
              v35 = sub_15A35F0(*(_WORD *)(v6 + 18) & 0x7FFF, v32, v34, 0);
              if ( v35 )
              {
                v65 = (__int64 *)v6;
                sub_38526A0(a1 + 136, (__int64 *)&v65)[1] = v35;
                ++*(_DWORD *)(a1 + 544);
                goto LABEL_38;
              }
            }
LABEL_33:
            v17 = sub_15FF0A0(v6, a2, v14, a4, a5);
            if ( !(_BYTE)v17 || *(_BYTE *)(*(_QWORD *)(v6 - 24) + 16LL) != 15 )
              goto LABEL_35;
            v36 = *(_QWORD *)(v6 - 48);
            if ( *(_BYTE *)(v36 + 16) == 17 )
            {
              v37 = *(_DWORD *)(v36 + 32);
              v59 = *(_QWORD *)(a1 + 56) & 0xFFFFFFFFFFFFFFF8LL;
              v38 = (_QWORD *)(v59 + 56);
              if ( (*(_QWORD *)(a1 + 56) & 4) != 0 )
              {
                if ( (unsigned __int8)sub_1560290(v38, v37, 32) )
                  goto LABEL_56;
                v39 = *(_QWORD *)(v59 - 24);
                if ( !*(_BYTE *)(v39 + 16) )
                {
LABEL_55:
                  v65 = *(__int64 **)(v39 + 112);
                  if ( (unsigned __int8)sub_1560290(&v65, v37, 32) )
                    goto LABEL_56;
                }
              }
              else
              {
                if ( (unsigned __int8)sub_1560290(v38, v37, 32) )
                  goto LABEL_56;
                v39 = *(_QWORD *)(v59 - 72);
                if ( !*(_BYTE *)(v39 + 16) )
                  goto LABEL_55;
              }
            }
            v43 = *(_DWORD *)(a1 + 192);
            if ( !v43 )
              goto LABEL_35;
            v44 = v43 - 1;
            v45 = *(_QWORD *)(a1 + 176);
            v46 = (v43 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
            v47 = *(_QWORD *)(v45 + 16LL * v46);
            if ( v47 != v36 )
            {
              v48 = 1;
              while ( v47 != -8 )
              {
                v46 = v44 & (v48 + v46);
                v47 = *(_QWORD *)(v45 + 16LL * v46);
                if ( v47 == v36 )
                  goto LABEL_56;
                ++v48;
              }
LABEL_35:
              v29 = *(_DWORD *)(a1 + 184);
              v65 = 0;
              v66 = -1;
              v67[0] = 0;
              v67[1] = 0;
              if ( v29 )
              {
                if ( *(_DWORD *)(a1 + 216) )
                {
                  LOBYTE(v30) = sub_384F1D0(a1, *(_QWORD *)(v6 - 48), &v60, &v65);
                  v17 = v30;
                  if ( (_BYTE)v30 )
                  {
                    if ( *(_BYTE *)(*(_QWORD *)(v6 - 24) + 16LL) == 15 )
                    {
                      *(_DWORD *)(v67[0] + 8) += 5;
                      *(_DWORD *)(a1 + 556) += 5;
LABEL_38:
                      if ( v64 > 0x40 && v63 )
                        j_j___libc_free_0_0(v63);
                      if ( v62 > 0x40 && v61 )
                        j_j___libc_free_0_0(v61);
                      return v17;
                    }
                    sub_384F170(a1, v67[0]);
                  }
                }
              }
              v17 = 0;
              goto LABEL_38;
            }
LABEL_56:
            v40 = *(unsigned __int16 *)(v6 + 18);
            v41 = *(_QWORD *)v6;
            BYTE1(v40) &= ~0x80u;
            if ( v40 == 33 )
              v42 = sub_15A0600(v41);
            else
              v42 = sub_15A0640(v41);
            v65 = (__int64 *)v6;
            sub_38526A0(a1 + 136, (__int64 *)&v65)[1] = v42;
            goto LABEL_38;
          }
          v54 = 1;
          while ( v27 != -8 )
          {
            v55 = v54 + 1;
            v14 = (unsigned int)a2 & (v54 + (_DWORD)v14);
            v26 = a4 + 32LL * (unsigned int)v14;
            v27 = *(_QWORD *)v26;
            if ( v56 == *(_QWORD *)v26 )
              goto LABEL_29;
            v54 = v55;
          }
        }
        v65 = 0;
        LODWORD(v67[0]) = 1;
        v66 = 0;
        v63 = 0;
        v64 = 1;
        goto LABEL_33;
      }
      v52 = 1;
      while ( v20 != -8 )
      {
        v53 = v52 + 1;
        v14 = (unsigned int)a2 & (v52 + (_DWORD)v14);
        v19 = a4 + 32LL * (unsigned int)v14;
        v20 = *(_QWORD *)v19;
        if ( v57 == *(_QWORD *)v19 )
          goto LABEL_17;
        v52 = v53;
      }
    }
    v61 = 0;
    v62 = 1;
    goto LABEL_33;
  }
  return v17;
}
