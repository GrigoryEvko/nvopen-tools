// Function: sub_2E2BE10
// Address: 0x2e2be10
//
__int64 __fastcall sub_2E2BE10(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r11
  __int64 v11; // rbx
  __int64 v12; // r14
  int v13; // r13d
  char v14; // al
  unsigned int *v15; // rax
  int v16; // r10d
  int v17; // ebx
  int i; // r14d
  unsigned int v19; // eax
  int v20; // edi
  unsigned int v21; // r13d
  unsigned __int64 v22; // rax
  __int64 *v23; // rdi
  unsigned int v24; // eax
  int v25; // esi
  int v27; // eax
  int v28; // r10d
  int v29; // r13d
  unsigned int *v30; // rdi
  unsigned int v31; // r13d
  __int64 v32; // rax
  __int64 v33; // rbx
  int v34; // r12d
  __int64 v35; // r15
  __int64 v36; // rax
  _QWORD *v37; // rax
  int v38; // r9d
  __int64 v39; // rax
  unsigned int v40; // esi
  int v41; // r15d
  int v42; // edi
  __int64 v43; // rsi
  unsigned int v44; // r15d
  int v45; // esi
  __int64 v46; // rax
  int v47; // eax
  unsigned int v48; // eax
  int v49; // r15d
  int v50; // esi
  unsigned int v51; // r15d
  int v52; // edi
  __int64 v53; // rax
  unsigned int v54; // eax
  int v55; // edi
  __int64 v56; // rsi
  unsigned int v57; // eax
  int v58; // edi
  int v59; // r11d
  __int64 v60; // rax
  unsigned int v61; // r10d
  unsigned int v62; // r10d
  __int64 v63; // [rsp+8h] [rbp-98h]
  __int64 v64; // [rsp+8h] [rbp-98h]
  __int64 v65; // [rsp+8h] [rbp-98h]
  __int64 v66; // [rsp+8h] [rbp-98h]
  __int64 v67; // [rsp+18h] [rbp-88h]
  unsigned int v69; // [rsp+2Ch] [rbp-74h]
  __int64 v70; // [rsp+30h] [rbp-70h] BYREF
  __int64 v71; // [rsp+38h] [rbp-68h]
  __int64 v72; // [rsp+40h] [rbp-60h]
  __int64 v73; // [rsp+48h] [rbp-58h]
  __int64 v74; // [rsp+50h] [rbp-50h] BYREF
  __int64 v75; // [rsp+58h] [rbp-48h]
  __int64 v76; // [rsp+60h] [rbp-40h]
  __int64 v77; // [rsp+68h] [rbp-38h]

  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)(a4 + 56);
  v67 = a4;
  v69 = *(_DWORD *)(a2 + 24);
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  if ( v7 == a4 + 48 )
    goto LABEL_25;
  v8 = a2;
  v9 = a4 + 48;
  while ( !*(_WORD *)(v7 + 68) || *(_WORD *)(v7 + 68) == 68 )
  {
    v29 = *(_DWORD *)(*(_QWORD *)(v7 + 32) + 8LL);
    if ( !(_DWORD)v73 )
    {
      ++v70;
      goto LABEL_127;
    }
    a6 = v71;
    a4 = ((_DWORD)v73 - 1) & (unsigned int)(37 * v29);
    v6 = a4;
    v30 = (unsigned int *)(v71 + 4 * a4);
    a5 = *v30;
    if ( v29 != (_DWORD)a5 )
    {
      v47 = 1;
      v6 = 0;
      while ( (_DWORD)a5 != -1 )
      {
        if ( (_DWORD)a5 == -2 && !v6 )
          v6 = (__int64)v30;
        v59 = v47 + 1;
        v60 = ((_DWORD)v73 - 1) & (unsigned int)(a4 + v47);
        v30 = (unsigned int *)(v71 + 4 * v60);
        a4 = (unsigned int)v60;
        a5 = *v30;
        if ( v29 == (_DWORD)a5 )
          goto LABEL_50;
        v47 = v59;
      }
      if ( !v6 )
        v6 = (__int64)v30;
      ++v70;
      a4 = (unsigned int)(v72 + 1);
      if ( 4 * (int)a4 < (unsigned int)(3 * v73) )
      {
        a5 = (unsigned int)v73 >> 3;
        if ( (int)v73 - HIDWORD(v72) - (int)a4 <= (unsigned int)a5 )
        {
          sub_2E29BA0((__int64)&v70, v73);
          if ( !(_DWORD)v73 )
          {
LABEL_186:
            LODWORD(v72) = v72 + 1;
            BUG();
          }
          v57 = (v73 - 1) & (37 * v29);
          v58 = 1;
          v56 = 0;
          a4 = (unsigned int)(v72 + 1);
          v6 = v71 + 4LL * v57;
          a5 = *(unsigned int *)v6;
          if ( v29 != (_DWORD)a5 )
          {
            while ( (_DWORD)a5 != -1 )
            {
              if ( (_DWORD)a5 == -2 && !v56 )
                v56 = v6;
              a6 = (unsigned int)(v58 + 1);
              v57 = (v73 - 1) & (v58 + v57);
              v6 = v71 + 4LL * v57;
              a5 = *(unsigned int *)v6;
              if ( v29 == (_DWORD)a5 )
                goto LABEL_108;
              ++v58;
            }
            goto LABEL_131;
          }
        }
        goto LABEL_108;
      }
LABEL_127:
      sub_2E29BA0((__int64)&v70, 2 * v73);
      if ( !(_DWORD)v73 )
        goto LABEL_186;
      v54 = (v73 - 1) & (37 * v29);
      a4 = (unsigned int)(v72 + 1);
      v6 = v71 + 4LL * v54;
      a5 = *(unsigned int *)v6;
      if ( v29 != (_DWORD)a5 )
      {
        v55 = 1;
        v56 = 0;
        while ( (_DWORD)a5 != -1 )
        {
          if ( !v56 && (_DWORD)a5 == -2 )
            v56 = v6;
          a6 = (unsigned int)(v55 + 1);
          v54 = (v73 - 1) & (v55 + v54);
          v6 = v71 + 4LL * v54;
          a5 = *(unsigned int *)v6;
          if ( v29 == (_DWORD)a5 )
            goto LABEL_108;
          ++v55;
        }
LABEL_131:
        if ( v56 )
          v6 = v56;
      }
LABEL_108:
      LODWORD(v72) = a4;
      if ( *(_DWORD *)v6 != -1 )
        --HIDWORD(v72);
      *(_DWORD *)v6 = v29;
    }
LABEL_50:
    v31 = 1;
    if ( (*(_DWORD *)(v7 + 40) & 0xFFFFFF) != 1 )
    {
      v32 = v8;
      v33 = v7;
      v34 = *(_DWORD *)(v7 + 40) & 0xFFFFFF;
      v35 = v32;
      do
      {
        while ( 1 )
        {
          v6 = *(_QWORD *)(v33 + 32);
          if ( v35 == *(_QWORD *)(v6 + 40LL * (v31 + 1) + 24) )
            break;
          v31 += 2;
          if ( v34 == v31 )
            goto LABEL_55;
        }
        v36 = v31;
        v31 += 2;
        v37 = (_QWORD *)sub_2E29D60(a1, *(_DWORD *)(v6 + 40 * v36 + 8), v6, a4, a5, a6);
        sub_FDE240(v37, v69);
      }
      while ( v34 != v31 );
LABEL_55:
      v7 = v33;
      v8 = v35;
    }
    if ( (*(_BYTE *)v7 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
        v7 = *(_QWORD *)(v7 + 8);
    }
    v7 = *(_QWORD *)(v7 + 8);
    if ( v9 == v7 )
      goto LABEL_25;
  }
  v10 = v9;
  if ( v9 != v7 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 32);
      v12 = v11 + 40LL * (*(_DWORD *)(v7 + 40) & 0xFFFFFF);
      if ( v11 != v12 )
        break;
LABEL_23:
      if ( (*(_BYTE *)v7 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
          v7 = *(_QWORD *)(v7 + 8);
      }
      v7 = *(_QWORD *)(v7 + 8);
      if ( v10 == v7 )
        goto LABEL_25;
    }
    while ( 2 )
    {
      if ( *(_BYTE *)v11 )
        goto LABEL_8;
      v13 = *(_DWORD *)(v11 + 8);
      if ( v13 >= 0 )
        goto LABEL_8;
      v14 = *(_BYTE *)(v11 + 3);
      if ( (v14 & 0x10) == 0 )
      {
        if ( (v14 & 0x40) == 0 )
          goto LABEL_8;
        if ( (_DWORD)v77 )
        {
          a5 = (unsigned int)(v77 - 1);
          a4 = (unsigned int)a5 & (37 * v13);
          v15 = (unsigned int *)(v75 + 4 * a4);
          v6 = *v15;
          if ( v13 == (_DWORD)v6 )
            goto LABEL_8;
          v16 = 1;
          a6 = 0;
          while ( (_DWORD)v6 != -1 )
          {
            if ( (_DWORD)v6 != -2 || a6 )
              v15 = (unsigned int *)a6;
            a6 = (unsigned int)(v16 + 1);
            a4 = (unsigned int)a5 & (v16 + (_DWORD)a4);
            v6 = *(unsigned int *)(v75 + 4LL * (unsigned int)a4);
            if ( v13 == (_DWORD)v6 )
              goto LABEL_8;
            ++v16;
            a6 = (__int64)v15;
            v15 = (unsigned int *)(v75 + 4LL * (unsigned int)a4);
          }
          if ( !a6 )
            a6 = (__int64)v15;
          ++v74;
          v6 = (unsigned int)(v76 + 1);
          if ( 4 * (int)v6 < (unsigned int)(3 * v77) )
          {
            a4 = (unsigned int)v77 >> 3;
            if ( (int)v77 - HIDWORD(v76) - (int)v6 <= (unsigned int)a4 )
            {
              v66 = v10;
              sub_2E29BA0((__int64)&v74, v77);
              if ( !(_DWORD)v77 )
              {
LABEL_187:
                LODWORD(v76) = v76 + 1;
                BUG();
              }
              a5 = v75;
              a4 = 1;
              v10 = v66;
              v51 = (v77 - 1) & (37 * v13);
              a6 = v75 + 4LL * v51;
              v52 = *(_DWORD *)a6;
              v6 = (unsigned int)(v76 + 1);
              v53 = 0;
              if ( v13 != *(_DWORD *)a6 )
              {
                while ( v52 != -1 )
                {
                  if ( v52 == -2 && !v53 )
                    v53 = a6;
                  v62 = a4 + 1;
                  a4 = ((_DWORD)v77 - 1) & (v51 + (unsigned int)a4);
                  a6 = v75 + 4 * a4;
                  v51 = a4;
                  v52 = *(_DWORD *)a6;
                  if ( v13 == *(_DWORD *)a6 )
                    goto LABEL_21;
                  a4 = v62;
                }
                if ( v53 )
                  a6 = v53;
              }
            }
LABEL_21:
            LODWORD(v76) = v6;
            if ( *(_DWORD *)a6 != -1 )
              --HIDWORD(v76);
LABEL_47:
            *(_DWORD *)a6 = v13;
LABEL_8:
            v11 += 40;
            if ( v12 == v11 )
              goto LABEL_23;
            continue;
          }
        }
        else
        {
          ++v74;
        }
        v65 = v10;
        sub_2E29BA0((__int64)&v74, 2 * v77);
        if ( !(_DWORD)v77 )
          goto LABEL_187;
        a5 = v75;
        v10 = v65;
        v48 = (v77 - 1) & (37 * v13);
        a6 = v75 + 4LL * v48;
        v49 = *(_DWORD *)a6;
        v6 = (unsigned int)(v76 + 1);
        if ( v13 != *(_DWORD *)a6 )
        {
          v50 = 1;
          a4 = 0;
          while ( v49 != -1 )
          {
            if ( !a4 && v49 == -2 )
              a4 = a6;
            v48 = (v77 - 1) & (v48 + v50);
            a6 = v75 + 4LL * v48;
            v49 = *(_DWORD *)a6;
            if ( v13 == *(_DWORD *)a6 )
              goto LABEL_21;
            ++v50;
          }
          if ( a4 )
            a6 = a4;
        }
        goto LABEL_21;
      }
      break;
    }
    if ( (_DWORD)v73 )
    {
      a5 = (unsigned int)(v73 - 1);
      a4 = (unsigned int)a5 & (37 * v13);
      v6 = v71 + 4 * a4;
      v27 = *(_DWORD *)v6;
      if ( v13 == *(_DWORD *)v6 )
        goto LABEL_8;
      v28 = 1;
      a6 = 0;
      while ( v27 != -1 )
      {
        if ( a6 || v27 != -2 )
          v6 = a6;
        a6 = (unsigned int)(v28 + 1);
        a4 = (unsigned int)a5 & (v28 + (_DWORD)a4);
        v27 = *(_DWORD *)(v71 + 4LL * (unsigned int)a4);
        if ( v13 == v27 )
          goto LABEL_8;
        ++v28;
        a6 = v6;
        v6 = v71 + 4LL * (unsigned int)a4;
      }
      if ( !a6 )
        a6 = v6;
      ++v70;
      v6 = (unsigned int)(v72 + 1);
      if ( 4 * (int)v6 < (unsigned int)(3 * v73) )
      {
        a4 = (unsigned int)v73 >> 3;
        if ( (int)v73 - HIDWORD(v72) - (int)v6 <= (unsigned int)a4 )
        {
          v64 = v10;
          sub_2E29BA0((__int64)&v70, v73);
          if ( !(_DWORD)v73 )
          {
LABEL_185:
            LODWORD(v72) = v72 + 1;
            BUG();
          }
          a5 = v71;
          a4 = 1;
          v10 = v64;
          v44 = (v73 - 1) & (37 * v13);
          a6 = v71 + 4LL * v44;
          v45 = *(_DWORD *)a6;
          v6 = (unsigned int)(v72 + 1);
          v46 = 0;
          if ( v13 != *(_DWORD *)a6 )
          {
            while ( v45 != -1 )
            {
              if ( !v46 && v45 == -2 )
                v46 = a6;
              v61 = a4 + 1;
              a4 = ((_DWORD)v73 - 1) & (v44 + (unsigned int)a4);
              a6 = v71 + 4 * a4;
              v44 = a4;
              v45 = *(_DWORD *)a6;
              if ( v13 == *(_DWORD *)a6 )
                goto LABEL_45;
              a4 = v61;
            }
            if ( v46 )
              a6 = v46;
          }
        }
        goto LABEL_45;
      }
    }
    else
    {
      ++v70;
    }
    v63 = v10;
    sub_2E29BA0((__int64)&v70, 2 * v73);
    if ( !(_DWORD)v73 )
      goto LABEL_185;
    a5 = v71;
    v10 = v63;
    a4 = ((_DWORD)v73 - 1) & (unsigned int)(37 * v13);
    a6 = v71 + 4 * a4;
    v41 = *(_DWORD *)a6;
    v6 = (unsigned int)(v72 + 1);
    if ( v13 != *(_DWORD *)a6 )
    {
      v42 = 1;
      v43 = 0;
      while ( v41 != -1 )
      {
        if ( !v43 && v41 == -2 )
          v43 = a6;
        a4 = ((_DWORD)v73 - 1) & (unsigned int)(a4 + v42);
        a6 = v71 + 4LL * (unsigned int)a4;
        v41 = *(_DWORD *)a6;
        if ( v13 == *(_DWORD *)a6 )
          goto LABEL_45;
        ++v42;
      }
      if ( v43 )
        a6 = v43;
    }
LABEL_45:
    LODWORD(v72) = v6;
    if ( *(_DWORD *)a6 != -1 )
      --HIDWORD(v72);
    goto LABEL_47;
  }
LABEL_25:
  v17 = *(_DWORD *)(a1[5].m128i_i64[1] + 64);
  if ( v17 )
  {
    for ( i = 0; i != v17; ++i )
    {
      while ( 1 )
      {
        v21 = i | 0x80000000;
        if ( (_DWORD)v73 )
        {
          a4 = (unsigned int)(v73 - 1);
          v19 = a4 & (37 * v21);
          v20 = *(_DWORD *)(v71 + 4LL * v19);
          if ( v21 == v20 )
            goto LABEL_28;
          a5 = 1;
          while ( v20 != -1 )
          {
            a6 = (unsigned int)(a5 + 1);
            v19 = a4 & (a5 + v19);
            v20 = *(_DWORD *)(v71 + 4LL * v19);
            if ( v21 == v20 )
              goto LABEL_28;
            a5 = (unsigned int)a6;
          }
        }
        v22 = sub_2E29D60(a1, v21, v6, a4, a5, a6);
        a5 = v75;
        v23 = (__int64 *)v22;
        if ( (_DWORD)v77 )
          break;
LABEL_64:
        a6 = *v23;
        if ( v23 == (__int64 *)*v23 )
          goto LABEL_28;
        a4 = *(unsigned int *)(v67 + 24);
        v39 = v23[3];
        if ( v23 == (__int64 *)v39 )
        {
          v39 = v23[1];
          v40 = (unsigned int)a4 >> 7;
          v23[3] = v39;
          a5 = *(unsigned int *)(v39 + 16);
          if ( (unsigned int)a4 >> 7 == (_DWORD)a5 )
          {
            if ( v23 == (__int64 *)v39 )
              goto LABEL_28;
LABEL_75:
            v6 = 1LL << a4;
            a4 = *(_QWORD *)(v39 + 8LL * (((unsigned int)a4 >> 6) & 1) + 24) & (1LL << a4);
            if ( a4 )
              goto LABEL_32;
            goto LABEL_28;
          }
        }
        else
        {
          a5 = *(unsigned int *)(v39 + 16);
          v40 = (unsigned int)a4 >> 7;
          if ( (_DWORD)a5 == (unsigned int)a4 >> 7 )
            goto LABEL_75;
        }
        if ( (unsigned int)a5 > v40 )
        {
          if ( a6 != v39 )
          {
            while ( 1 )
            {
              v39 = *(_QWORD *)(v39 + 8);
              if ( a6 == v39 )
                break;
              if ( *(_DWORD *)(v39 + 16) <= v40 )
                goto LABEL_73;
            }
          }
          v23[3] = v39;
        }
        else
        {
          if ( v23 == (__int64 *)v39 )
          {
LABEL_134:
            v23[3] = v39;
            goto LABEL_28;
          }
          while ( (unsigned int)a5 < v40 )
          {
            v39 = *(_QWORD *)v39;
            if ( v23 == (__int64 *)v39 )
              goto LABEL_134;
            a5 = *(unsigned int *)(v39 + 16);
          }
LABEL_73:
          v23[3] = v39;
          if ( v23 == (__int64 *)v39 )
            goto LABEL_28;
        }
        if ( *(_DWORD *)(v39 + 16) == v40 )
          goto LABEL_75;
LABEL_28:
        if ( ++i == v17 )
          goto LABEL_33;
      }
      a4 = (unsigned int)(v77 - 1);
      v24 = a4 & (37 * v21);
      v25 = *(_DWORD *)(v75 + 4LL * v24);
      if ( v21 != v25 )
      {
        v38 = 1;
        while ( v25 != -1 )
        {
          v24 = a4 & (v38 + v24);
          v25 = *(_DWORD *)(v75 + 4LL * v24);
          if ( v21 == v25 )
            goto LABEL_32;
          ++v38;
        }
        goto LABEL_64;
      }
LABEL_32:
      sub_FDE240(v23, v69);
    }
  }
LABEL_33:
  sub_C7D6A0(v75, 4LL * (unsigned int)v77, 4);
  return sub_C7D6A0(v71, 4LL * (unsigned int)v73, 4);
}
