// Function: sub_DB77A0
// Address: 0xdb77a0
//
__int64 __fastcall sub_DB77A0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r12d
  __int64 v5; // r14
  int v6; // r12d
  unsigned int v7; // eax
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 result; // rax
  int v11; // edi
  unsigned int v12; // eax
  __int64 v13; // rdi
  int v14; // esi
  int v15; // edx
  _QWORD *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // edx
  _QWORD *v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rsi
  char v24; // r8
  unsigned int v25; // esi
  __int64 v26; // rcx
  __int64 v27; // r8
  int v28; // r10d
  unsigned int v29; // edx
  __int64 *v30; // r13
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // r12
  unsigned int v34; // r8d
  __int64 v35; // r9
  int v36; // ebx
  __int64 *v37; // r11
  __int64 v38; // rsi
  unsigned int v39; // edx
  __int64 *v40; // rcx
  __int64 v41; // rdi
  int v42; // eax
  __int64 *v43; // rax
  int v44; // edi
  int v45; // edi
  _QWORD *v46; // rdi
  __int64 v47; // rsi
  __int64 *v48; // rbx
  __int64 *v49; // r14
  __int64 v50; // r8
  __int64 *v51; // rcx
  __int64 v52; // rdi
  unsigned int v53; // esi
  int v54; // edx
  int v55; // ecx
  int v56; // esi
  _QWORD *v57; // r9
  int v58; // r11d
  int v59; // r11d
  __int64 *v60; // r10
  int v61; // eax
  int v62; // [rsp+4h] [rbp-10Ch]
  __int64 v63; // [rsp+8h] [rbp-108h]
  char v64; // [rsp+8h] [rbp-108h]
  char v65; // [rsp+8h] [rbp-108h]
  __int64 v66; // [rsp+10h] [rbp-100h] BYREF
  __int64 v67; // [rsp+18h] [rbp-F8h] BYREF
  void *v68; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v69; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v70; // [rsp+38h] [rbp-D8h]
  __int64 v71; // [rsp+40h] [rbp-D0h]
  _QWORD *v72; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v73; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v74; // [rsp+68h] [rbp-A8h]
  void *v75; // [rsp+80h] [rbp-90h] BYREF
  _QWORD v76[2]; // [rsp+88h] [rbp-88h] BYREF
  __int64 v77; // [rsp+98h] [rbp-78h]
  __int64 *v78; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v79[2]; // [rsp+B8h] [rbp-58h] BYREF
  __int64 v80; // [rsp+C8h] [rbp-48h]
  __int64 v81; // [rsp+D0h] [rbp-40h]
  __int64 v82; // [rsp+D8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 152);
  v67 = a2;
  v66 = a3;
  if ( !v4 )
    goto LABEL_12;
  v5 = *(_QWORD *)(a1 + 136);
  v6 = v4 - 1;
  sub_D982A0(&v78, -4096, 0);
  a2 = v67;
  v7 = v6 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
  v8 = v5 + 48LL * v7;
  v9 = *(_QWORD *)(v8 + 24);
  if ( v67 != v9 )
  {
    v11 = 1;
    while ( v80 != v9 )
    {
      v7 = v6 & (v11 + v7);
      v8 = v5 + 48LL * v7;
      v9 = *(_QWORD *)(v8 + 24);
      if ( v67 == v9 )
        goto LABEL_3;
      ++v11;
    }
    if ( v80 && v80 != -8192 && v80 != -4096 )
    {
      v78 = (__int64 *)&unk_49DB368;
      sub_BD60C0(v79);
      a2 = v67;
    }
LABEL_12:
    sub_D982A0(&v68, a2, a1);
    v79[1] = 0;
    v79[0] = v69 & 6;
    v80 = v70;
    if ( v70 != -4096 && v70 != 0 && v70 != -8192 )
      sub_BD6050(v79, v69 & 0xFFFFFFFFFFFFFFF8LL);
    v78 = (__int64 *)&unk_49DE910;
    v81 = v71;
    v82 = v66;
    v12 = *(_DWORD *)(a1 + 152);
    if ( v12 )
    {
      v62 = *(_DWORD *)(a1 + 152);
      v63 = *(_QWORD *)(a1 + 136);
      sub_D982A0(&v72, -4096, 0);
      sub_D982A0(&v75, -8192, 0);
      v20 = (v62 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
      v21 = (_QWORD *)(v63 + 48LL * v20);
      v22 = v21[3];
      if ( v80 == v22 )
      {
        v23 = v77;
        v24 = 1;
      }
      else
      {
        v23 = v77;
        v57 = (_QWORD *)(v63 + 48LL * ((v62 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4))));
        v58 = 1;
        v21 = 0;
        while ( v74 != v22 )
        {
          if ( v77 == v22 && !v21 )
            v21 = v57;
          v20 = (v62 - 1) & (v58 + v20);
          v57 = (_QWORD *)(v63 + 48LL * v20);
          v22 = v57[3];
          if ( v80 == v22 )
          {
            v21 = (_QWORD *)(v63 + 48LL * v20);
            v24 = 1;
            goto LABEL_35;
          }
          ++v58;
        }
        v24 = 0;
        if ( !v21 )
          v21 = v57;
      }
LABEL_35:
      v75 = &unk_49DB368;
      if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
      {
        v64 = v24;
        sub_BD60C0(v76);
        v24 = v64;
      }
      v72 = &unk_49DB368;
      if ( v74 != 0 && v74 != -4096 && v74 != -8192 )
      {
        v65 = v24;
        sub_BD60C0(&v73);
        v24 = v65;
      }
      if ( v24 )
      {
        v19 = v80;
        goto LABEL_43;
      }
      v55 = *(_DWORD *)(a1 + 144);
      v12 = *(_DWORD *)(a1 + 152);
      v72 = v21;
      v13 = a1 + 128;
      ++*(_QWORD *)(a1 + 128);
      v15 = v55 + 1;
      if ( 4 * (v55 + 1) < 3 * v12 )
      {
        if ( v12 - (v15 + *(_DWORD *)(a1 + 148)) > v12 >> 3 )
          goto LABEL_19;
        v14 = v12;
LABEL_18:
        sub_DB30A0(v13, v14);
        sub_D9F8F0(v13, (__int64)&v78, &v72);
        v15 = *(_DWORD *)(a1 + 144) + 1;
LABEL_19:
        *(_DWORD *)(a1 + 144) = v15;
        sub_D982A0(&v75, -4096, 0);
        v16 = v72;
        v17 = v77;
        if ( v77 != v72[3] )
          --*(_DWORD *)(a1 + 148);
        v75 = &unk_49DB368;
        if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
          sub_BD60C0(v76);
        v18 = v16[3];
        v19 = v80;
        if ( v18 != v80 )
        {
          if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
          {
            sub_BD60C0(v16 + 1);
            v19 = v80;
          }
          v16[3] = v19;
          if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
            sub_BD6050(v16 + 1, v79[0] & 0xFFFFFFFFFFFFFFF8LL);
          v19 = v80;
        }
        v16[4] = v81;
        v16[5] = v82;
LABEL_43:
        v78 = (__int64 *)&unk_49DB368;
        if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
          sub_BD60C0(v79);
        v68 = &unk_49DB368;
        if ( v70 != -4096 && v70 != 0 && v70 != -8192 )
          sub_BD60C0(&v69);
        v25 = *(_DWORD *)(a1 + 120);
        if ( v25 )
        {
          v26 = v66;
          v27 = *(_QWORD *)(a1 + 104);
          v28 = 1;
          v29 = (v25 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          v30 = (__int64 *)(v27 + 88LL * v29);
          v31 = 0;
          v32 = *v30;
          if ( *v30 == v66 )
          {
LABEL_51:
            result = *((unsigned int *)v30 + 6);
            v33 = (__int64)(v30 + 1);
            if ( (_DWORD)result )
            {
              v34 = *((_DWORD *)v30 + 8);
              if ( v34 )
              {
                v35 = v30[2];
                v36 = 1;
                v37 = 0;
                v38 = v67;
                v39 = (v34 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
                v40 = (__int64 *)(v35 + 8LL * v39);
                v41 = *v40;
                if ( v67 == *v40 )
                  return result;
                while ( v41 != -4096 )
                {
                  if ( v41 == -8192 && !v37 )
                    v37 = v40;
                  v39 = (v34 - 1) & (v36 + v39);
                  v40 = (__int64 *)(v35 + 8LL * v39);
                  v41 = *v40;
                  if ( v67 == *v40 )
                    return result;
                  ++v36;
                }
                if ( !v37 )
                  v37 = v40;
                v42 = result + 1;
                v78 = v37;
                ++v30[1];
                if ( 4 * v42 < 3 * v34 )
                {
                  if ( v34 - (v42 + *((_DWORD *)v30 + 7)) > v34 >> 3 )
                  {
LABEL_59:
                    *((_DWORD *)v30 + 6) = v42;
                    v43 = v78;
                    if ( *v78 != -4096 )
                      --*((_DWORD *)v30 + 7);
                    *v43 = v38;
                    return sub_94F890((__int64)(v30 + 5), v67);
                  }
                  v56 = v34;
LABEL_95:
                  sub_CE2A30((__int64)(v30 + 1), v56);
                  sub_DA5B20((__int64)(v30 + 1), &v67, &v78);
                  v38 = v67;
                  v42 = *((_DWORD *)v30 + 6) + 1;
                  goto LABEL_59;
                }
              }
              else
              {
                v78 = 0;
                ++v30[1];
              }
              v56 = 2 * v34;
              goto LABEL_95;
            }
            goto LABEL_75;
          }
          while ( v32 != -4096 )
          {
            if ( !v31 && v32 == -8192 )
              v31 = v30;
            v29 = (v25 - 1) & (v28 + v29);
            v30 = (__int64 *)(v27 + 88LL * v29);
            v32 = *v30;
            if ( v66 == *v30 )
              goto LABEL_51;
            ++v28;
          }
          v44 = *(_DWORD *)(a1 + 112);
          if ( !v31 )
            v31 = v30;
          ++*(_QWORD *)(a1 + 96);
          v45 = v44 + 1;
          v78 = v31;
          if ( 4 * v45 < 3 * v25 )
          {
            if ( v25 - *(_DWORD *)(a1 + 116) - v45 > v25 >> 3 )
            {
LABEL_72:
              *(_DWORD *)(a1 + 112) = v45;
              if ( *v31 != -4096 )
                --*(_DWORD *)(a1 + 116);
              *v31 = v26;
              v33 = (__int64)(v31 + 1);
              v31[5] = (__int64)(v31 + 7);
              v31[6] = 0x400000000LL;
              *(_OWORD *)(v31 + 1) = 0;
              *(_OWORD *)(v31 + 3) = 0;
              *(_OWORD *)(v31 + 7) = 0;
              *(_OWORD *)(v31 + 9) = 0;
LABEL_75:
              v46 = *(_QWORD **)(v33 + 32);
              v47 = (__int64)&v46[*(unsigned int *)(v33 + 40)];
              result = (__int64)sub_D91230(v46, v47, &v67);
              if ( v47 != result )
                return result;
              sub_94F890(v33 + 32, v67);
              result = *(unsigned int *)(v33 + 40);
              if ( (unsigned int)result <= 4 )
                return result;
              v48 = *(__int64 **)(v33 + 32);
              v49 = &v48[result];
              while ( 1 )
              {
                v53 = *(_DWORD *)(v33 + 24);
                if ( !v53 )
                  break;
                v50 = *(_QWORD *)(v33 + 8);
                result = (v53 - 1) & (((unsigned int)*v48 >> 9) ^ ((unsigned int)*v48 >> 4));
                v51 = (__int64 *)(v50 + 8 * result);
                v52 = *v51;
                if ( *v48 != *v51 )
                {
                  v59 = 1;
                  v60 = 0;
                  while ( v52 != -4096 )
                  {
                    if ( !v60 && v52 == -8192 )
                      v60 = v51;
                    result = (v53 - 1) & (v59 + (_DWORD)result);
                    v51 = (__int64 *)(v50 + 8LL * (unsigned int)result);
                    v52 = *v51;
                    if ( *v48 == *v51 )
                      goto LABEL_79;
                    ++v59;
                  }
                  if ( !v60 )
                    v60 = v51;
                  v78 = v60;
                  v61 = *(_DWORD *)(v33 + 16);
                  ++*(_QWORD *)v33;
                  v54 = v61 + 1;
                  if ( 4 * (v61 + 1) < 3 * v53 )
                  {
                    if ( v53 - *(_DWORD *)(v33 + 20) - v54 <= v53 >> 3 )
                    {
LABEL_83:
                      sub_CE2A30(v33, v53);
                      sub_DA5B20(v33, v48, &v78);
                      v54 = *(_DWORD *)(v33 + 16) + 1;
                    }
                    *(_DWORD *)(v33 + 16) = v54;
                    result = (__int64)v78;
                    if ( *v78 != -4096 )
                      --*(_DWORD *)(v33 + 20);
                    *(_QWORD *)result = *v48;
                    goto LABEL_79;
                  }
LABEL_82:
                  v53 *= 2;
                  goto LABEL_83;
                }
LABEL_79:
                if ( v49 == ++v48 )
                  return result;
              }
              v78 = 0;
              ++*(_QWORD *)v33;
              goto LABEL_82;
            }
LABEL_98:
            sub_DA6240(a1 + 96, v25);
            sub_D9E560(a1 + 96, &v66, &v78);
            v26 = v66;
            v45 = *(_DWORD *)(a1 + 112) + 1;
            v31 = v78;
            goto LABEL_72;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 96);
          v78 = 0;
        }
        v25 *= 2;
        goto LABEL_98;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 128);
      v13 = a1 + 128;
      v72 = 0;
    }
    v14 = 2 * v12;
    goto LABEL_18;
  }
LABEL_3:
  v78 = (__int64 *)&unk_49DB368;
  if ( v80 && v80 != -4096 && v80 != -8192 )
    sub_BD60C0(v79);
  result = *(_QWORD *)(a1 + 136) + 48LL * *(unsigned int *)(a1 + 152);
  if ( v8 == result )
  {
    a2 = v67;
    goto LABEL_12;
  }
  return result;
}
