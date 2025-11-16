// Function: sub_1101600
// Address: 0x1101600
//
unsigned __int8 *__fastcall sub_1101600(_QWORD *a1, __int64 a2)
{
  unsigned __int8 *v4; // r15
  __int64 v5; // r14
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned int v8; // ebx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 *v11; // r9
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rbx
  int v15; // edx
  __int64 *v16; // r13
  __int64 v17; // r12
  unsigned __int8 *result; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  int v22; // eax
  int v23; // eax
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r10
  __int64 *v27; // r13
  __int64 v28; // r15
  __int64 v29; // r9
  __int64 v30; // rdx
  int v31; // r15d
  __int64 v32; // rbx
  __int64 v33; // r13
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // r10
  __int64 v37; // rbx
  __int64 *v38; // rdi
  __int64 v39; // rax
  int v40; // esi
  __int64 v41; // rdi
  int v42; // esi
  __int64 v43; // rbx
  _BYTE *v44; // rax
  __int64 v45; // rdx
  int v46; // r12d
  __int64 v47; // r12
  __int64 v48; // r13
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 *v51; // rbx
  _BYTE *v52; // rax
  __int64 v53; // r15
  unsigned int v54; // eax
  unsigned int v55; // r11d
  __int64 v56; // r15
  __int64 v57; // rdx
  int v58; // r14d
  __int64 v59; // r14
  __int64 v60; // rbx
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // [rsp+0h] [rbp-B0h]
  __int64 v64; // [rsp+0h] [rbp-B0h]
  __int64 v65; // [rsp+0h] [rbp-B0h]
  int v66; // [rsp+8h] [rbp-A8h]
  __int64 v67; // [rsp+8h] [rbp-A8h]
  __int64 v68; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v69; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v70; // [rsp+8h] [rbp-A8h]
  unsigned int v71; // [rsp+8h] [rbp-A8h]
  int v72; // [rsp+8h] [rbp-A8h]
  __int64 v73; // [rsp+18h] [rbp-98h]
  _BYTE v74[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v75; // [rsp+40h] [rbp-70h]
  _BYTE v76[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v77; // [rsp+70h] [rbp-40h]

  v4 = *(unsigned __int8 **)(a2 - 32);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *((_QWORD *)v4 + 1);
  v7 = v6;
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v6 + 16);
  v63 = *((_QWORD *)v4 + 1);
  v8 = *(_DWORD *)(v7 + 8) >> 8;
  v66 = sub_BCB060(*(_QWORD *)(a2 + 8));
  if ( v66 != sub_AE2980(a1[11], v8)[1] )
  {
    v67 = a1[11];
    v12 = sub_BD5C60(a2);
    v13 = (__int64 *)sub_AE4420(v67, v12, v8);
    v14 = (__int64)v13;
    v15 = *(unsigned __int8 *)(v63 + 8);
    if ( (unsigned int)(v15 - 17) <= 1 )
    {
      BYTE4(v73) = (_BYTE)v15 == 18;
      LODWORD(v73) = *(_DWORD *)(v63 + 32);
      v14 = sub_BCE1B0(v13, v73);
    }
    v16 = (__int64 *)a1[4];
    v75 = 257;
    if ( v14 == *((_QWORD *)v4 + 1) )
    {
      v17 = (__int64)v4;
    }
    else
    {
      v17 = (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int8 *, __int64))(*(_QWORD *)v16[10] + 120LL))(
              v16[10],
              47,
              v4,
              v14);
      if ( !v17 )
      {
        v77 = 257;
        v17 = sub_B51D30(47, (__int64)v4, v14, (__int64)v76, 0, 0);
        if ( (unsigned __int8)sub_920620(v17) )
        {
          v30 = v16[12];
          v31 = *((_DWORD *)v16 + 26);
          if ( v30 )
            sub_B99FD0(v17, 3u, v30);
          sub_B45150(v17, v31);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v16[11] + 16LL))(
          v16[11],
          v17,
          v74,
          v16[7],
          v16[8]);
        v32 = *v16;
        v33 = *v16 + 16LL * *((unsigned int *)v16 + 2);
        while ( v33 != v32 )
        {
          v34 = *(_QWORD *)(v32 + 8);
          v35 = *(_DWORD *)v32;
          v32 += 16;
          sub_B99FD0(v17, v35, v34);
        }
      }
    }
    v77 = 257;
    return (unsigned __int8 *)sub_B522D0(v17, v5, 0, (__int64)v76, 0, 0);
  }
  v19 = *((_QWORD *)v4 + 2);
  v20 = *v4;
  if ( v19 && !*(_QWORD *)(v19 + 8) && (_BYTE)v20 == 85 )
  {
    v9 = *((_QWORD *)v4 - 4);
    if ( v9 )
    {
      if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *((_QWORD *)v4 + 10) && *(_DWORD *)(v9 + 36) == 299 )
      {
        v9 = -32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF);
        v36 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        if ( v36 )
        {
          v9 = 32 * (1LL - (*((_DWORD *)v4 + 1) & 0x7FFFFFF));
          v37 = *(_QWORD *)&v4[v9];
          if ( v37 )
          {
            if ( *(_QWORD *)(v37 + 8) == v5 )
            {
              v38 = (__int64 *)a1[4];
              v77 = 257;
              v75 = 257;
              v39 = sub_10FF770(v38, 47, v36, v5, (__int64)v74, 0, v73, 0);
              return (unsigned __int8 *)sub_B504D0(28, v39, v37, (__int64)v76, 0, 0);
            }
          }
        }
      }
    }
LABEL_15:
    if ( *(_QWORD *)(v19 + 8) || (_BYTE)v20 != 91 )
      return sub_11005E0(a1, (unsigned __int8 *)a2, v20, v9, v10, v11);
    if ( (v4[7] & 0x40) != 0 )
      v20 = *((_QWORD *)v4 - 1);
    else
      v20 = (__int64)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
    v21 = *(_QWORD *)v20;
    v22 = **(unsigned __int8 **)v20;
    if ( (unsigned __int8)v22 > 0x1Cu )
    {
      v23 = v22 - 29;
    }
    else
    {
      if ( (_BYTE)v22 != 5 )
        return sub_11005E0(a1, (unsigned __int8 *)a2, v20, v9, v10, v11);
      v23 = *(unsigned __int16 *)(v21 + 2);
    }
    if ( v23 == 48 )
    {
      v24 = *(_QWORD *)sub_986520(v21);
      if ( v24 )
      {
        v25 = sub_986520((__int64)v4);
        v26 = *(_QWORD *)(v25 + 32);
        if ( v26 )
        {
          v64 = *(_QWORD *)(v25 + 64);
          if ( v64 )
          {
            if ( v5 == *(_QWORD *)(v24 + 8) )
            {
              v27 = (__int64 *)a1[4];
              v75 = 257;
              if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
                v5 = **(_QWORD **)(v5 + 16);
              if ( v5 == *(_QWORD *)(v26 + 8) )
              {
                v28 = v26;
              }
              else
              {
                v68 = v26;
                v28 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v27[10] + 120LL))(
                        v27[10],
                        47,
                        v26,
                        v5);
                if ( !v28 )
                {
                  v77 = 257;
                  v28 = sub_B51D30(47, v68, v5, (__int64)v76, 0, 0);
                  if ( (unsigned __int8)sub_920620(v28) )
                  {
                    v45 = v27[12];
                    v46 = *((_DWORD *)v27 + 26);
                    if ( v45 )
                      sub_B99FD0(v28, 3u, v45);
                    sub_B45150(v28, v46);
                  }
                  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v27[11] + 16LL))(
                    v27[11],
                    v28,
                    v74,
                    v27[7],
                    v27[8]);
                  v47 = *v27;
                  v48 = *v27 + 16LL * *((unsigned int *)v27 + 2);
                  while ( v48 != v47 )
                  {
                    v49 = *(_QWORD *)(v47 + 8);
                    v50 = *(_DWORD *)v47;
                    v47 += 16;
                    sub_B99FD0(v28, v50, v49);
                  }
                }
              }
              v77 = 257;
              result = (unsigned __int8 *)sub_BD2C40(72, 3u);
              if ( result )
              {
                v69 = result;
                sub_B4DFA0((__int64)result, v24, v28, v64, (__int64)v76, v29, 0, 0);
                return v69;
              }
              return result;
            }
          }
        }
      }
    }
    return sub_11005E0(a1, (unsigned __int8 *)a2, v20, v9, v10, v11);
  }
  if ( (unsigned __int8)v20 > 0x1Cu )
  {
    if ( (_BYTE)v20 != 63 )
      goto LABEL_14;
  }
  else if ( (_BYTE)v20 != 5 || *((_WORD *)v4 + 1) != 34 )
  {
    goto LABEL_14;
  }
  if ( !v19 )
    return sub_11005E0(a1, (unsigned __int8 *)a2, v20, v9, v10, v11);
  if ( *(_QWORD *)(v19 + 8) )
  {
LABEL_14:
    if ( !v19 )
      return sub_11005E0(a1, (unsigned __int8 *)a2, v20, v9, v10, v11);
    goto LABEL_15;
  }
  v9 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  v40 = *(unsigned __int8 *)v9;
  if ( (_BYTE)v40 != 20 )
  {
    v41 = *(_QWORD *)(v9 + 16);
    if ( v41 && !*(_QWORD *)(v41 + 8) )
    {
      if ( (unsigned __int8)v40 > 0x1Cu )
      {
        v42 = v40 - 29;
      }
      else
      {
        if ( (_BYTE)v40 != 5 )
          goto LABEL_14;
        v42 = *(unsigned __int16 *)(v9 + 2);
      }
      if ( v42 == 48 )
      {
        if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
          v9 = *(_QWORD *)(v9 - 8);
        else
          v9 -= 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
        v43 = *(_QWORD *)v9;
        if ( *(_QWORD *)v9 )
        {
          if ( v5 == *(_QWORD *)(v43 + 8) )
          {
            v44 = sub_F20BF0((__int64)a1, (__int64)v4, 0);
            v77 = 257;
            v70 = (unsigned __int8 *)sub_B504D0(13, v43, (__int64)v44, (__int64)v76, 0, 0);
            sub_B447F0(v70, (v4[1] & 8) != 0);
            return v70;
          }
        }
      }
    }
    goto LABEL_14;
  }
  v75 = 257;
  v51 = (__int64 *)a1[4];
  v52 = sub_F20BF0((__int64)a1, (__int64)v4, 0);
  v53 = *((_QWORD *)v52 + 1);
  v65 = (__int64)v52;
  v71 = sub_BCB060(v53);
  v54 = sub_BCB060(v5);
  v55 = (v71 <= v54) + 38;
  if ( v5 == v53 )
  {
    v56 = v65;
  }
  else
  {
    v72 = (v71 <= v54) + 38;
    v56 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v51[10] + 120LL))(
            v51[10],
            v55,
            v65,
            v5);
    if ( !v56 )
    {
      v77 = 257;
      v56 = sub_B51D30(v72, v65, v5, (__int64)v76, 0, 0);
      if ( (unsigned __int8)sub_920620(v56) )
      {
        v57 = v51[12];
        v58 = *((_DWORD *)v51 + 26);
        if ( v57 )
          sub_B99FD0(v56, 3u, v57);
        sub_B45150(v56, v58);
      }
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v51[11] + 16LL))(
        v51[11],
        v56,
        v74,
        v51[7],
        v51[8]);
      v59 = *v51 + 16LL * *((unsigned int *)v51 + 2);
      if ( *v51 != v59 )
      {
        v60 = *v51;
        do
        {
          v61 = *(_QWORD *)(v60 + 8);
          v62 = *(_DWORD *)v60;
          v60 += 16;
          sub_B99FD0(v56, v62, v61);
        }
        while ( v59 != v60 );
      }
    }
  }
  return sub_F162A0((__int64)a1, a2, v56);
}
