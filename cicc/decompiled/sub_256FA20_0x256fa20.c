// Function: sub_256FA20
// Address: 0x256fa20
//
__int64 __fastcall sub_256FA20(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *v4; // rax
  _QWORD *v5; // r12
  __int64 v6; // rdx
  _QWORD *v7; // rbx
  unsigned __int64 *v9; // r15
  char *v10; // r14
  unsigned __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 *v15; // r10
  _QWORD *v16; // rax
  unsigned int v17; // esi
  int v18; // eax
  unsigned __int64 *v19; // r14
  int v20; // eax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // rcx
  unsigned __int64 *v28; // rsi
  unsigned __int64 *v29; // rdi
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rsi
  unsigned __int64 *v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // r9
  __int64 v35; // r15
  __int64 v36; // r12
  unsigned int v37; // esi
  unsigned __int64 *v38; // r8
  int v39; // eax
  _QWORD *v40; // rsi
  unsigned int v41; // esi
  __int64 v42; // r10
  __int64 v43; // rdx
  unsigned int v44; // eax
  unsigned __int64 *v45; // r9
  unsigned __int64 v46; // rdi
  int v47; // ecx
  unsigned __int64 *v48; // r8
  int v49; // eax
  __int64 v50; // rdi
  __int64 v51; // rax
  unsigned __int64 v52; // r14
  __int64 v53; // rdi
  unsigned __int64 *v54; // [rsp-D8h] [rbp-D8h]
  _QWORD *v55; // [rsp-C8h] [rbp-C8h]
  unsigned __int64 *v56; // [rsp-C0h] [rbp-C0h]
  unsigned __int64 *v57; // [rsp-C0h] [rbp-C0h]
  __int64 v58; // [rsp-B0h] [rbp-B0h]
  unsigned __int64 *v59; // [rsp-A0h] [rbp-A0h] BYREF
  _QWORD v60[2]; // [rsp-98h] [rbp-98h] BYREF
  __int64 v61; // [rsp-88h] [rbp-88h]
  _QWORD v62[4]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v63; // [rsp-58h] [rbp-58h] BYREF
  __int64 v64; // [rsp-50h] [rbp-50h]
  __int64 v65; // [rsp-48h] [rbp-48h]

  v2 = *(unsigned int *)(a1 + 124);
  if ( *(_DWORD *)(a1 + 128) == (_DWORD)v2 )
    return 1;
  v4 = *(_QWORD **)(a1 + 112);
  if ( !*(_BYTE *)(a1 + 132) )
    v2 = *(unsigned int *)(a1 + 120);
  v5 = &v4[v2];
  if ( v4 != v5 )
  {
    while ( 1 )
    {
      v6 = *v4;
      v7 = v4;
      if ( *v4 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v5 == ++v4 )
        return 0;
    }
    if ( v5 != v4 )
    {
      v9 = v60;
      v58 = a2 + 2688;
      while ( 1 )
      {
        v61 = v6;
        v60[0] = 4;
        v60[1] = 0;
        if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
          sub_BD73F0((__int64)v9);
        if ( *(_DWORD *)(a2 + 2704) )
          break;
        v10 = *(char **)(a2 + 2720);
        v11 = sub_2538140(v10, (__int64)&v10[24 * *(unsigned int *)(a2 + 2728)], (__int64)v9);
        if ( v15 == v11 )
        {
          v31 = v13 + 1;
          v32 = v9;
          if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 2732) )
          {
            v50 = a2 + 2720;
            if ( v10 > (char *)v9 || v15 <= v9 )
            {
              sub_D6B130(v50, v31, (__int64)v9, v12, v13, v14);
              v32 = v9;
              LODWORD(v14) = *(_DWORD *)(a2 + 2728);
              v15 = (unsigned __int64 *)(*(_QWORD *)(a2 + 2720) + 24LL * (unsigned int)v14);
            }
            else
            {
              sub_D6B130(v50, v31, (__int64)v9, v12, v13, v14);
              v51 = *(_QWORD *)(a2 + 2720);
              v14 = *(unsigned int *)(a2 + 2728);
              v32 = (unsigned __int64 *)(v51 + (char *)v9 - v10);
              v15 = (unsigned __int64 *)(v51 + 24 * v14);
            }
          }
          if ( v15 )
          {
            *v15 = 4;
            v33 = v32[2];
            v15[1] = 0;
            v15[2] = v33;
            if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
              sub_BD6050(v15, *v32 & 0xFFFFFFFFFFFFFFF8LL);
            LODWORD(v14) = *(_DWORD *)(a2 + 2728);
          }
          v34 = (unsigned int)(v14 + 1);
          *(_DWORD *)(a2 + 2728) = v34;
          if ( (unsigned int)v34 > 0x10 )
          {
            v55 = v5;
            v54 = v9;
            v35 = *(_QWORD *)(a2 + 2720);
            v36 = v35 + 24 * v34;
            while ( 1 )
            {
              v37 = *(_DWORD *)(a2 + 2712);
              if ( !v37 )
                break;
              v62[0] = 4;
              v41 = v37 - 1;
              v42 = *(_QWORD *)(a2 + 2696);
              v62[1] = 0;
              v62[2] = -4096;
              v63 = 4;
              v64 = 0;
              v65 = -8192;
              v43 = *(_QWORD *)(v35 + 16);
              v44 = v41 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
              v45 = (unsigned __int64 *)(v42 + 24LL * v44);
              v46 = v45[2];
              if ( v43 == v46 )
              {
LABEL_54:
                v35 += 24;
                sub_D68D70(&v63);
                sub_D68D70(v62);
                if ( v36 == v35 )
                  goto LABEL_52;
              }
              else
              {
                v47 = 1;
                v48 = 0;
                while ( v46 != -4096 )
                {
                  if ( v48 || v46 != -8192 )
                    v45 = v48;
                  v44 = v41 & (v47 + v44);
                  v46 = *(_QWORD *)(v42 + 24LL * v44 + 16);
                  if ( v43 == v46 )
                    goto LABEL_54;
                  ++v47;
                  v48 = v45;
                  v45 = (unsigned __int64 *)(v42 + 24LL * v44);
                }
                if ( v48 )
                  v57 = v48;
                else
                  v57 = v45;
                sub_D68D70(&v63);
                sub_D68D70(v62);
                v37 = *(_DWORD *)(a2 + 2712);
                v38 = v57;
                v49 = *(_DWORD *)(a2 + 2704);
                ++*(_QWORD *)(a2 + 2688);
                v62[0] = v57;
                v39 = v49 + 1;
                if ( 4 * v39 < 3 * v37 )
                {
                  if ( v37 - (v39 + *(_DWORD *)(a2 + 2708)) > v37 >> 3 )
                    goto LABEL_49;
                  goto LABEL_48;
                }
LABEL_47:
                v37 *= 2;
LABEL_48:
                sub_2517BE0(v58, v37);
                sub_25116B0(v58, v35, v62);
                v38 = (unsigned __int64 *)v62[0];
                v39 = *(_DWORD *)(a2 + 2704) + 1;
LABEL_49:
                *(_DWORD *)(a2 + 2704) = v39;
                v63 = 4;
                v64 = 0;
                v65 = -4096;
                if ( v38[2] != -4096 )
                  --*(_DWORD *)(a2 + 2708);
                v56 = v38;
                sub_D68D70(&v63);
                v40 = (_QWORD *)v35;
                v35 += 24;
                sub_2538AB0(v56, v40);
                if ( v36 == v35 )
                {
LABEL_52:
                  v5 = v55;
                  v9 = v54;
                  goto LABEL_15;
                }
              }
            }
            ++*(_QWORD *)(a2 + 2688);
            v62[0] = 0;
            goto LABEL_47;
          }
        }
LABEL_15:
        if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
          sub_BD60C0(v9);
        v16 = v7 + 1;
        if ( v7 + 1 == v5 )
          return 0;
        v6 = *v16;
        for ( ++v7; *v16 >= 0xFFFFFFFFFFFFFFFELL; v7 = v16 )
        {
          if ( v5 == ++v16 )
            return 0;
          v6 = *v16;
        }
        if ( v5 == v7 )
          return 0;
      }
      if ( (unsigned __int8)sub_25116B0(v58, (__int64)v9, &v59) )
        goto LABEL_15;
      v17 = *(_DWORD *)(a2 + 2712);
      v18 = *(_DWORD *)(a2 + 2704);
      v19 = v59;
      ++*(_QWORD *)(a2 + 2688);
      v20 = v18 + 1;
      v62[0] = v19;
      if ( 4 * v20 >= 3 * v17 )
      {
        v17 *= 2;
      }
      else if ( v17 - *(_DWORD *)(a2 + 2708) - v20 > v17 >> 3 )
      {
LABEL_27:
        *(_DWORD *)(a2 + 2704) = v20;
        v63 = 4;
        v64 = 0;
        v65 = -4096;
        if ( v19[2] != -4096 )
          --*(_DWORD *)(a2 + 2708);
        sub_D68D70(&v63);
        sub_2538AB0(v19, v9);
        v23 = *(unsigned int *)(a2 + 2728);
        v24 = *(unsigned int *)(a2 + 2732);
        v25 = v23 + 1;
        v26 = *(_DWORD *)(a2 + 2728);
        if ( v23 + 1 > v24 )
        {
          v52 = *(_QWORD *)(a2 + 2720);
          v53 = a2 + 2720;
          if ( v52 > (unsigned __int64)v9 || (unsigned __int64)v9 >= v52 + 24 * v23 )
          {
            sub_D6B130(v53, v25, v23, v24, v21, v22);
            v23 = *(unsigned int *)(a2 + 2728);
            v27 = *(_QWORD *)(a2 + 2720);
            v28 = v9;
            v26 = *(_DWORD *)(a2 + 2728);
          }
          else
          {
            sub_D6B130(v53, v25, v23, v24, v21, v22);
            v27 = *(_QWORD *)(a2 + 2720);
            v23 = *(unsigned int *)(a2 + 2728);
            v28 = (unsigned __int64 *)((char *)v9 + v27 - v52);
            v26 = *(_DWORD *)(a2 + 2728);
          }
        }
        else
        {
          v27 = *(_QWORD *)(a2 + 2720);
          v28 = v9;
        }
        v29 = (unsigned __int64 *)(v27 + 24 * v23);
        if ( v29 )
        {
          *v29 = 4;
          v30 = v28[2];
          v29[1] = 0;
          v29[2] = v30;
          if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
            sub_BD6050(v29, *v28 & 0xFFFFFFFFFFFFFFF8LL);
          v26 = *(_DWORD *)(a2 + 2728);
        }
        *(_DWORD *)(a2 + 2728) = v26 + 1;
        goto LABEL_15;
      }
      sub_2517BE0(v58, v17);
      sub_25116B0(v58, (__int64)v9, v62);
      v19 = (unsigned __int64 *)v62[0];
      v20 = *(_DWORD *)(a2 + 2704) + 1;
      goto LABEL_27;
    }
  }
  return 0;
}
