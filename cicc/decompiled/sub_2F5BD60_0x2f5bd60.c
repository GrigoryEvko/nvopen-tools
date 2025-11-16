// Function: sub_2F5BD60
// Address: 0x2f5bd60
//
__int64 __fastcall sub_2F5BD60(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 i; // rbx
  __int64 v20; // r13
  __int64 v21; // rcx
  unsigned int v22; // edx
  __int64 v23; // rax
  _DWORD *v24; // rax
  __int64 v25; // rcx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  _QWORD *v31; // rdi
  __int64 *v32; // rsi
  __int64 v33; // r8
  __int64 v34; // r9
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rdi
  unsigned int v39; // edx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rax
  _QWORD *v45; // r13
  _QWORD *v46; // rbx
  __int64 v47; // r8
  unsigned int v48; // eax
  _QWORD *v49; // rdi
  __int64 v50; // rcx
  unsigned int v51; // esi
  int v52; // r15d
  int v53; // r15d
  __int64 v54; // r11
  unsigned int v55; // edx
  _QWORD *v56; // r10
  __int64 v57; // rdi
  int v58; // eax
  int v59; // r11d
  int v60; // eax
  int v61; // r15d
  int v62; // r15d
  __int64 v63; // r11
  _QWORD *v64; // rcx
  int v65; // esi
  unsigned int v66; // edx
  __int64 v67; // rdi
  int v68; // esi
  int v69; // r9d
  __int64 v70; // rsi
  int v71; // r9d
  __int64 v72; // rsi
  __int64 v73; // [rsp+0h] [rbp-C0h]
  __int64 v74; // [rsp+8h] [rbp-B8h]
  unsigned int v75; // [rsp+14h] [rbp-ACh]
  unsigned __int64 v76; // [rsp+20h] [rbp-A0h]
  __int64 v77; // [rsp+28h] [rbp-98h]
  unsigned int v80; // [rsp+3Ch] [rbp-84h]
  __int16 *v81; // [rsp+40h] [rbp-80h]
  __int64 v82; // [rsp+48h] [rbp-78h]
  __int64 v83; // [rsp+58h] [rbp-68h] BYREF
  _BYTE v84[96]; // [rsp+60h] [rbp-60h] BYREF

  v5 = a1;
  v8 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 56LL) + 16LL * (*(_DWORD *)(a3 + 112) & 0x7FFFFFFF));
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_QWORD *)(v9 + 8);
  v76 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
  v81 = (__int16 *)(*(_QWORD *)(v9 + 56) + 2LL * (*(_DWORD *)(v10 + 24LL * a2 + 16) >> 12));
  v75 = a2 - 1;
  v80 = *(_DWORD *)(v10 + 24LL * a2 + 16) & 0xFFF;
  v82 = a5 + 88;
  while ( 1 )
  {
    if ( !v81 )
      return 1;
    v11 = sub_2E21610(*(_QWORD *)(v5 + 40), a3, v80);
    v16 = v11;
    if ( *(_BYTE *)(v11 + 161) )
    {
      v12 = *(unsigned int *)(v11 + 120);
      v17 = (unsigned int)v12;
      if ( (unsigned int)qword_5024088 >= (unsigned int)v12 )
      {
        if ( (_DWORD)qword_5024088 != (_DWORD)v12 )
          goto LABEL_8;
        if ( !(_BYTE)qword_5023FA8 )
          break;
        goto LABEL_7;
      }
    }
    sub_2E1AC90(v16, qword_5024088, v12, v13, v14, v15);
    if ( (unsigned int)qword_5024088 <= *(_DWORD *)(v16 + 120) && !(_BYTE)qword_5023FA8 )
      break;
    if ( *(_BYTE *)(v16 + 161) )
    {
      LODWORD(v12) = *(_DWORD *)(v16 + 120);
LABEL_7:
      v17 = (unsigned int)v12;
      goto LABEL_8;
    }
    sub_2E1AC90(v16, 0xFFFFFFFF, v27, v28, v29, v30);
    v17 = *(unsigned int *)(v16 + 120);
LABEL_8:
    v18 = *(_QWORD *)(v16 + 112);
    if ( v18 != v18 + 8 * v17 )
    {
      v77 = *(_QWORD *)(v16 + 112);
      for ( i = v18 + 8 * v17; v77 != i; i -= 8 )
      {
        v20 = *(_QWORD *)(i - 8);
        v21 = *(_QWORD *)(v5 + 920);
        v22 = *(_DWORD *)(v20 + 112);
        v83 = v20;
        v23 = v22 & 0x7FFFFFFF;
        if ( *(_DWORD *)(v21 + 8 * v23) == 6 )
        {
          v38 = *(_QWORD *)(v5 + 16);
          if ( v76 == (*(_QWORD *)(*(_QWORD *)(v38 + 56) + 16 * v23) & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v39 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 24) + 32LL) + 4 * v23);
            if ( a2 != v39 && v75 <= 0x3FFFFFFE && v39 - 1 <= 0x3FFFFFFE )
            {
              if ( (unsigned __int8)sub_E92070(*(_QWORD *)(v5 + 8), a2, v39) )
              {
                v20 = v83;
                v22 = *(_DWORD *)(v83 + 112);
                goto LABEL_11;
              }
              v38 = *(_QWORD *)(v5 + 16);
            }
            if ( !(unsigned __int8)sub_2F4D680(v38, *(_DWORD *)(a3 + 112)) )
              return 0;
            v20 = v83;
            if ( (unsigned __int8)sub_2F4D680(v38, *(_DWORD *)(v83 + 112)) )
              return 0;
          }
        }
LABEL_11:
        if ( *(_QWORD *)(a5 + 120) )
        {
          v36 = *(_QWORD *)(a5 + 96);
          if ( v36 )
          {
            v37 = v82;
            do
            {
              if ( *(_DWORD *)(v36 + 32) < v22 )
              {
                v36 = *(_QWORD *)(v36 + 24);
              }
              else
              {
                v37 = v36;
                v36 = *(_QWORD *)(v36 + 16);
              }
            }
            while ( v36 );
            if ( v82 != v37 && *(_DWORD *)(v37 + 32) <= v22 )
              return 0;
          }
        }
        else
        {
          v24 = *(_DWORD **)a5;
          v25 = *(_QWORD *)a5 + 4LL * *(unsigned int *)(a5 + 8);
          if ( *(_QWORD *)a5 != v25 )
          {
            while ( *v24 != v22 )
            {
              if ( (_DWORD *)v25 == ++v24 )
                goto LABEL_25;
            }
            if ( (_DWORD *)v25 != v24 )
              return 0;
          }
        }
LABEL_25:
        if ( *(_DWORD *)(a4 + 16) )
        {
          sub_2F5B510((__int64)v84, a4, &v83);
          if ( v84[32] )
          {
            v42 = *(unsigned int *)(a4 + 40);
            v43 = v83;
            if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
            {
              sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v42 + 1, 8u, v40, v41);
              v42 = *(unsigned int *)(a4 + 40);
            }
            *(_QWORD *)(*(_QWORD *)(a4 + 32) + 8 * v42) = v43;
            ++*(_DWORD *)(a4 + 40);
          }
        }
        else
        {
          v31 = *(_QWORD **)(a4 + 32);
          v32 = &v31[*(unsigned int *)(a4 + 40)];
          if ( v32 == sub_2F4C750(v31, (__int64)v32, &v83) )
          {
            if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
            {
              sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v33 + 1, 8u, v33, v34);
              v32 = (__int64 *)(*(_QWORD *)(a4 + 32) + 8LL * *(unsigned int *)(a4 + 40));
            }
            *v32 = v20;
            v44 = (unsigned int)(*(_DWORD *)(a4 + 40) + 1);
            *(_DWORD *)(a4 + 40) = v44;
            if ( (unsigned int)v44 > 4 )
            {
              v74 = i;
              v73 = v5;
              v45 = *(_QWORD **)(a4 + 32);
              v46 = &v45[v44];
              while ( 1 )
              {
                v51 = *(_DWORD *)(a4 + 24);
                if ( !v51 )
                  break;
                v47 = *(_QWORD *)(a4 + 8);
                v48 = (v51 - 1) & (((unsigned int)*v45 >> 9) ^ ((unsigned int)*v45 >> 4));
                v49 = (_QWORD *)(v47 + 8LL * v48);
                v50 = *v49;
                if ( *v45 != *v49 )
                {
                  v59 = 1;
                  v56 = 0;
                  while ( v50 != -4096 )
                  {
                    if ( v50 != -8192 || v56 )
                      v49 = v56;
                    v48 = (v51 - 1) & (v59 + v48);
                    v50 = *(_QWORD *)(v47 + 8LL * v48);
                    if ( *v45 == v50 )
                      goto LABEL_51;
                    ++v59;
                    v56 = v49;
                    v49 = (_QWORD *)(v47 + 8LL * v48);
                  }
                  v60 = *(_DWORD *)(a4 + 16);
                  if ( !v56 )
                    v56 = v49;
                  ++*(_QWORD *)a4;
                  v58 = v60 + 1;
                  if ( 4 * v58 < 3 * v51 )
                  {
                    if ( v51 - *(_DWORD *)(a4 + 20) - v58 <= v51 >> 3 )
                    {
                      sub_2F5B340(a4, v51);
                      v61 = *(_DWORD *)(a4 + 24);
                      if ( !v61 )
                      {
LABEL_94:
                        ++*(_DWORD *)(a4 + 16);
                        BUG();
                      }
                      v62 = v61 - 1;
                      v63 = *(_QWORD *)(a4 + 8);
                      v64 = 0;
                      v65 = 1;
                      v66 = v62 & (((unsigned int)*v45 >> 9) ^ ((unsigned int)*v45 >> 4));
                      v56 = (_QWORD *)(v63 + 8LL * v66);
                      v67 = *v56;
                      v58 = *(_DWORD *)(a4 + 16) + 1;
                      if ( *v45 != *v56 )
                      {
                        while ( v67 != -4096 )
                        {
                          if ( !v64 && v67 == -8192 )
                            v64 = v56;
                          v71 = v65 + 1;
                          v72 = v62 & (v66 + v65);
                          v56 = (_QWORD *)(v63 + 8 * v72);
                          v66 = v72;
                          v67 = *v56;
                          if ( *v45 == *v56 )
                            goto LABEL_56;
                          v65 = v71;
                        }
LABEL_74:
                        if ( v64 )
                          v56 = v64;
                      }
                    }
LABEL_56:
                    *(_DWORD *)(a4 + 16) = v58;
                    if ( *v56 != -4096 )
                      --*(_DWORD *)(a4 + 20);
                    *v56 = *v45;
                    goto LABEL_51;
                  }
LABEL_54:
                  sub_2F5B340(a4, 2 * v51);
                  v52 = *(_DWORD *)(a4 + 24);
                  if ( !v52 )
                    goto LABEL_94;
                  v53 = v52 - 1;
                  v54 = *(_QWORD *)(a4 + 8);
                  v55 = v53 & (((unsigned int)*v45 >> 9) ^ ((unsigned int)*v45 >> 4));
                  v56 = (_QWORD *)(v54 + 8LL * v55);
                  v57 = *v56;
                  v58 = *(_DWORD *)(a4 + 16) + 1;
                  if ( *v45 != *v56 )
                  {
                    v68 = 1;
                    v64 = 0;
                    while ( v57 != -4096 )
                    {
                      if ( !v64 && v57 == -8192 )
                        v64 = v56;
                      v69 = v68 + 1;
                      v70 = v53 & (v55 + v68);
                      v56 = (_QWORD *)(v54 + 8 * v70);
                      v55 = v70;
                      v57 = *v56;
                      if ( *v45 == *v56 )
                        goto LABEL_56;
                      v68 = v69;
                    }
                    goto LABEL_74;
                  }
                  goto LABEL_56;
                }
LABEL_51:
                if ( v46 == ++v45 )
                {
                  i = v74;
                  v5 = v73;
                  goto LABEL_27;
                }
              }
              ++*(_QWORD *)a4;
              goto LABEL_54;
            }
          }
        }
LABEL_27:
        ;
      }
    }
    v35 = *v81;
    v80 += v35;
    ++v81;
    if ( !(_WORD)v35 )
      return 1;
  }
  *(_BYTE *)(v5 + 984) |= 2u;
  return 0;
}
