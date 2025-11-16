// Function: sub_118FA60
// Address: 0x118fa60
//
_QWORD *__fastcall sub_118FA60(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  void *v7; // r11
  unsigned int v8; // ebx
  char *v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // r8d
  char v16; // al
  __int64 v17; // r14
  unsigned int **v18; // rdi
  __int64 v19; // r12
  int *v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  _DWORD *v32; // rsi
  int v33; // r8d
  int *v34; // r10
  char v35; // al
  unsigned int **v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned int **v41; // rdi
  __int64 v42; // r12
  unsigned int **v43; // rdi
  __int64 v44; // r12
  void *v45; // [rsp+0h] [rbp-B0h]
  void *v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+10h] [rbp-A0h]
  int *v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  __int64 v51; // [rsp+18h] [rbp-98h]
  void *v52; // [rsp+20h] [rbp-90h]
  __int64 v53; // [rsp+20h] [rbp-90h]
  __int64 v54; // [rsp+20h] [rbp-90h]
  __int64 v55; // [rsp+28h] [rbp-88h]
  void *v56; // [rsp+28h] [rbp-88h]
  void *v57; // [rsp+28h] [rbp-88h]
  int *v58; // [rsp+28h] [rbp-88h]
  void *v59; // [rsp+28h] [rbp-88h]
  __int64 v60; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v61; // [rsp+38h] [rbp-78h]
  char *v62; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v63; // [rsp+48h] [rbp-68h]
  char *v64; // [rsp+50h] [rbp-60h] BYREF
  __int64 v65; // [rsp+58h] [rbp-58h]
  __int16 v66; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 - 96);
  v4 = *(_QWORD *)(a2 - 64);
  v55 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v3 == 85 )
  {
    v5 = *(_QWORD *)(v3 - 32);
    if ( v5 )
    {
      if ( !*(_BYTE *)v5 && *(_QWORD *)(v5 + 24) == *(_QWORD *)(v3 + 80) && *(_DWORD *)(v5 + 36) == 402 )
      {
        v24 = *(_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
        if ( v24 )
        {
          v64 = (char *)a1;
          v65 = a2;
          if ( *(_BYTE *)v4 == 85
            && (v29 = *(_QWORD *)(v4 - 32)) != 0
            && !*(_BYTE *)v29
            && *(_QWORD *)(v29 + 24) == *(_QWORD *)(v4 + 80)
            && *(_DWORD *)(v29 + 36) == 402
            && *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)) )
          {
            v30 = *(_QWORD *)(v3 + 16);
            if ( *(_BYTE *)v55 == 85 )
            {
              v37 = *(_QWORD *)(v55 - 32);
              if ( v37 )
              {
                if ( !*(_BYTE *)v37 && *(_QWORD *)(v37 + 24) == *(_QWORD *)(v55 + 80) && *(_DWORD *)(v37 + 36) == 402 )
                {
                  v38 = *(_QWORD *)(v55 - 32LL * (*(_DWORD *)(v55 + 4) & 0x7FFFFFF));
                  if ( v38 )
                  {
                    if ( v30 && !*(_QWORD *)(v30 + 8) )
                      return sub_1178BF0(
                               (__int64 *)&v64,
                               v24,
                               *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)),
                               v38);
                    v39 = *(_QWORD *)(v4 + 16);
                    if ( v39 )
                    {
                      if ( !*(_QWORD *)(v39 + 8) )
                        return sub_1178BF0(
                                 (__int64 *)&v64,
                                 v24,
                                 *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)),
                                 v38);
                    }
                    v40 = *(_QWORD *)(v55 + 16);
                    if ( v40 )
                    {
                      if ( !*(_QWORD *)(v40 + 8) )
                        return sub_1178BF0(
                                 (__int64 *)&v64,
                                 v24,
                                 *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)),
                                 v38);
                    }
                  }
                }
              }
            }
            if ( v30 && !*(_QWORD *)(v30 + 8) || (v31 = *(_QWORD *)(v4 + 16)) != 0 && !*(_QWORD *)(v31 + 8) )
            {
              v54 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
              if ( sub_9B7DA0((char *)v55, 0xFFFFFFFF, 0) )
                return sub_1178BF0((__int64 *)&v64, v24, v54, v55);
            }
          }
          else if ( sub_9B7DA0((char *)v4, 0xFFFFFFFF, 0) && *(_BYTE *)v55 == 85 )
          {
            v25 = *(_QWORD *)(v55 - 32);
            if ( v25 )
            {
              if ( !*(_BYTE *)v25 && *(_QWORD *)(v25 + 24) == *(_QWORD *)(v55 + 80) && *(_DWORD *)(v25 + 36) == 402 )
              {
                v26 = *(_QWORD *)(v55 - 32LL * (*(_DWORD *)(v55 + 4) & 0x7FFFFFF));
                if ( v26 )
                {
                  v27 = *(_QWORD *)(v3 + 16);
                  if ( v27 )
                  {
                    if ( !*(_QWORD *)(v27 + 8) )
                      return sub_1178BF0((__int64 *)&v64, v24, v4, v26);
                  }
                  v28 = *(_QWORD *)(v55 + 16);
                  if ( v28 )
                  {
                    if ( !*(_QWORD *)(v28 + 8) )
                      return sub_1178BF0((__int64 *)&v64, v24, v4, v26);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v6 = *(_QWORD *)(a2 + 8);
  v7 = 0;
  if ( *(_BYTE *)(v6 + 8) == 17 )
  {
    v8 = *(_DWORD *)(v6 + 32);
    v61 = v8;
    if ( v8 > 0x40 )
    {
      sub_C43690((__int64)&v60, 0, 0);
      v63 = v8;
      sub_C43690((__int64)&v62, -1, 1);
      LODWORD(v65) = v63;
      if ( v63 > 0x40 )
      {
        sub_C43780((__int64)&v64, (const void **)&v62);
LABEL_11:
        v10 = sub_11A3F30(a1, a2, &v64, &v60, 0, 0);
        v7 = (void *)v10;
        if ( (unsigned int)v65 > 0x40 && v64 )
        {
          v52 = (void *)v10;
          j_j___libc_free_0_0(v64);
          v7 = v52;
        }
        if ( v7 )
        {
          if ( (void *)a2 != v7 )
            v7 = sub_F162A0(a1, a2, (__int64)v7);
          goto LABEL_17;
        }
        v12 = *(_QWORD *)(v4 + 16);
        if ( !v12 )
          goto LABEL_27;
        if ( *(_QWORD *)(v12 + 8) )
          goto LABEL_27;
        if ( *(_BYTE *)v4 != 92 )
          goto LABEL_27;
        v51 = *(_QWORD *)(v4 - 64);
        if ( !v51 )
          goto LABEL_27;
        v47 = *(_QWORD *)(v4 - 32);
        if ( !v47 )
          goto LABEL_27;
        v32 = (_DWORD *)(*(_QWORD *)(v4 + 72) + 4LL * *(unsigned int *)(v4 + 80));
        v53 = *(unsigned int *)(v4 + 80);
        if ( v32 != sub_1178410(*(_DWORD **)(v4 + 72), (__int64)v32, &dword_3F92D40) )
          goto LABEL_27;
        if ( v33 != *(_DWORD *)(*(_QWORD *)(v51 + 8) + 32LL) )
          goto LABEL_27;
        v45 = v7;
        v49 = v34;
        v35 = sub_B4EEA0(v34, v53, v33);
        v7 = v45;
        if ( !v35 )
          goto LABEL_27;
        if ( v55 == v51 )
        {
          v41 = *(unsigned int ***)(a1 + 32);
          v64 = "sel";
          v66 = 259;
          v42 = sub_B36550(v41, v3, v47, v51, (__int64)&v64, a2);
          v66 = 257;
          v7 = sub_BD2C40(112, unk_3F1FE60);
          if ( v7 )
          {
            v22 = v42;
            v21 = v53;
            v23 = v51;
            v20 = v49;
            goto LABEL_40;
          }
          goto LABEL_17;
        }
        if ( v55 == v47 )
        {
          v17 = v55;
          v36 = *(unsigned int ***)(a1 + 32);
          v64 = "sel";
          v66 = 259;
          v19 = sub_B36550(v36, v3, v51, v55, (__int64)&v64, a2);
          v66 = 257;
          v7 = sub_BD2C40(112, unk_3F1FE60);
          if ( v7 )
          {
            v20 = v49;
LABEL_39:
            v21 = v53;
            v22 = v17;
            v23 = v19;
LABEL_40:
            v59 = v7;
            sub_B4E9E0((__int64)v7, v23, v22, v20, v21, (__int64)&v64, 0, 0);
            v7 = v59;
          }
        }
        else
        {
LABEL_27:
          v13 = *(_QWORD *)(v55 + 16);
          if ( !v13 )
            goto LABEL_17;
          if ( *(_QWORD *)(v13 + 8) )
            goto LABEL_17;
          if ( *(_BYTE *)v55 != 92 )
            goto LABEL_17;
          v50 = *(_QWORD *)(v55 - 64);
          if ( !v50 )
            goto LABEL_17;
          v48 = *(_QWORD *)(v55 - 32);
          if ( !v48 )
            goto LABEL_17;
          v14 = *(unsigned int *)(v55 + 80);
          v58 = *(int **)(v55 + 72);
          v53 = v14;
          if ( &v58[v14] != sub_1178410(v58, (__int64)&v58[v14], &dword_3F92D40) )
            goto LABEL_17;
          if ( v15 != *(_DWORD *)(*(_QWORD *)(v50 + 8) + 32LL) )
            goto LABEL_17;
          v46 = v7;
          v16 = sub_B4EEA0(v58, v53, v15);
          v7 = v46;
          if ( !v16 )
            goto LABEL_17;
          if ( v4 != v50 )
          {
            v7 = 0;
            if ( v4 != v48 )
              goto LABEL_17;
            v17 = v48;
            v18 = *(unsigned int ***)(a1 + 32);
            v64 = "sel";
            v66 = 259;
            v19 = sub_B36550(v18, v3, v48, v50, (__int64)&v64, a2);
            v66 = 257;
            v7 = sub_BD2C40(112, unk_3F1FE60);
            if ( !v7 )
              goto LABEL_17;
            v20 = v58;
            goto LABEL_39;
          }
          v43 = *(unsigned int ***)(a1 + 32);
          v64 = "sel";
          v66 = 259;
          v44 = sub_B36550(v43, v3, v50, v48, (__int64)&v64, a2);
          v66 = 257;
          v7 = sub_BD2C40(112, unk_3F1FE60);
          if ( v7 )
          {
            v20 = v58;
            v22 = v44;
            v21 = v53;
            v23 = v50;
            goto LABEL_40;
          }
        }
LABEL_17:
        if ( v63 > 0x40 && v62 )
        {
          v56 = v7;
          j_j___libc_free_0_0(v62);
          v7 = v56;
        }
        if ( v61 > 0x40 && v60 )
        {
          v57 = v7;
          j_j___libc_free_0_0(v60);
          return v57;
        }
        return v7;
      }
    }
    else
    {
      v60 = 0;
      v63 = v8;
      v9 = (char *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v8);
      LODWORD(v65) = v8;
      if ( !v8 )
        v9 = 0;
      v62 = v9;
    }
    v64 = v62;
    goto LABEL_11;
  }
  return v7;
}
