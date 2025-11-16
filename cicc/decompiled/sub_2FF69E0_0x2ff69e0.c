// Function: sub_2FF69E0
// Address: 0x2ff69e0
//
_WORD *__fastcall sub_2FF69E0(
        __int64 a1,
        _QWORD *a2,
        int a3,
        _QWORD *a4,
        unsigned int a5,
        unsigned int *a6,
        unsigned int *a7)
{
  _QWORD *v9; // r13
  __int64 v10; // rcx
  unsigned __int64 v11; // rbx
  __int64 v12; // rcx
  unsigned int *v13; // rcx
  __int64 v14; // rax
  unsigned int v15; // eax
  _WORD *v16; // rsi
  __int64 v17; // r11
  __int64 v18; // r15
  __int64 v19; // r14
  _QWORD *v20; // r13
  __int64 v21; // rbx
  _QWORD *v22; // r8
  __int64 v23; // r13
  __int64 v24; // rax
  unsigned __int16 *v25; // r15
  unsigned int v26; // r14d
  __int64 v27; // r12
  __int64 v28; // rax
  unsigned int v29; // edi
  __int64 v30; // r11
  _WORD *v33; // r13
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 v36; // r11
  unsigned int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rsi
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v44; // [rsp+0h] [rbp-C0h]
  _QWORD *v45; // [rsp+8h] [rbp-B8h]
  _WORD *v46; // [rsp+10h] [rbp-B0h]
  unsigned int v49; // [rsp+2Ch] [rbp-94h]
  __int64 v50; // [rsp+30h] [rbp-90h]
  _QWORD *v51; // [rsp+38h] [rbp-88h]
  _QWORD *v52; // [rsp+38h] [rbp-88h]
  unsigned int v53; // [rsp+40h] [rbp-80h]
  int v54; // [rsp+44h] [rbp-7Ch]
  _WORD *v55; // [rsp+48h] [rbp-78h]
  unsigned int v56; // [rsp+50h] [rbp-70h]
  __int64 v57; // [rsp+60h] [rbp-60h]
  _QWORD *v58; // [rsp+68h] [rbp-58h]
  unsigned __int64 v59; // [rsp+68h] [rbp-58h]
  __int64 v60; // [rsp+70h] [rbp-50h] BYREF
  char v61; // [rsp+78h] [rbp-48h]
  __int64 v62; // [rsp+80h] [rbp-40h] BYREF
  char v63; // [rsp+88h] [rbp-38h]

  v9 = a2;
  v45 = a4;
  v53 = a5;
  v10 = *(unsigned int *)(*(_QWORD *)(a1 + 312)
                        + 16LL
                        * (*(unsigned __int16 *)(*a2 + 24LL)
                         + *(_DWORD *)(a1 + 328)
                         * (unsigned int)((__int64)(*(_QWORD *)(a1 + 288) - *(_QWORD *)(a1 + 280)) >> 3)));
  v61 = 0;
  v60 = v10;
  v11 = sub_CA1930(&v60);
  v12 = *(unsigned int *)(*(_QWORD *)(a1 + 312)
                        + 16LL
                        * (*(unsigned __int16 *)(*a4 + 24LL)
                         + *(_DWORD *)(a1 + 328)
                         * (unsigned int)((__int64)(*(_QWORD *)(a1 + 288) - *(_QWORD *)(a1 + 280)) >> 3)));
  v63 = 0;
  v62 = v12;
  if ( v11 < sub_CA1930(&v62) )
  {
    v13 = a7;
    a7 = a6;
    a6 = v13;
    v53 = a3;
    v9 = a4;
    a3 = a5;
    v45 = a2;
  }
  v14 = *(unsigned int *)(*(_QWORD *)(a1 + 312)
                        + 16LL
                        * (*(unsigned __int16 *)(*v9 + 24LL)
                         + *(_DWORD *)(a1 + 328)
                         * (unsigned int)((__int64)(*(_QWORD *)(a1 + 288) - *(_QWORD *)(a1 + 280)) >> 3)));
  v63 = 0;
  v62 = v14;
  v15 = sub_CA1930(&v62);
  v16 = (_WORD *)v9[2];
  v17 = *(_QWORD *)(a1 + 288);
  v18 = *(_QWORD *)(a1 + 280);
  v56 = v15;
  v55 = v16;
  if ( v16 )
  {
    v19 = v9[1];
    v20 = (_QWORD *)a1;
    v46 = v16;
    v44 = 4LL * (((unsigned int)((v17 - v18) >> 3) + 31) >> 5);
    v49 = 0;
    v54 = a3;
    v55 = 0;
    while ( 1 )
    {
      v21 = v45[1];
      if ( v45[2] )
      {
        v22 = v20;
        v23 = v18;
        v57 = 4LL * (((unsigned int)((v17 - v18) >> 3) + 31) >> 5);
        v24 = v19;
        v25 = (unsigned __int16 *)v45[2];
        v26 = 0;
        v27 = v24;
        while ( 1 )
        {
          v28 = 0;
          v29 = 0;
          v30 = (v17 - v23) >> 3;
          if ( (_DWORD)v30 )
          {
            while ( !(*(_DWORD *)(v21 + v28) & *(_DWORD *)(v27 + v28)) )
            {
              v29 += 32;
              v28 += 4;
              if ( (unsigned int)v30 <= v29 )
                goto LABEL_21;
            }
            __asm { tzcnt   edx, edx }
            v33 = *(_WORD **)(v23 + 8LL * (_EDX + v29));
            if ( v33 )
            {
              v58 = v22;
              v34 = *(unsigned int *)(v22[39]
                                    + 16LL
                                    * (*((_DWORD *)v22 + 82) * (_DWORD)v30
                                     + (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v33 + 24LL)));
              v63 = 0;
              v62 = v34;
              v35 = sub_CA1930(&v62);
              v36 = v56;
              v22 = v58;
              if ( v35 >= v56 )
              {
                v37 = v53;
                if ( v26 )
                {
                  v37 = v26;
                  if ( v53 )
                  {
                    v37 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))(*v58 + 296LL))(v58, v26);
                    v36 = v56;
                    v22 = v58;
                  }
                }
                if ( v54 == v37 )
                {
                  if ( !v55 )
                    goto LABEL_19;
                  v50 = v36;
                  v51 = v22;
                  v38 = *(unsigned int *)(v22[39]
                                        + 16LL
                                        * (*(unsigned __int16 *)(*(_QWORD *)v33 + 24LL)
                                         + *((_DWORD *)v22 + 82) * (unsigned int)((__int64)(v22[36] - v22[35]) >> 3)));
                  v61 = 0;
                  v60 = v38;
                  v59 = sub_CA1930(&v60);
                  v39 = *(unsigned int *)(v51[39]
                                        + 16LL
                                        * (*(unsigned __int16 *)(*(_QWORD *)v55 + 24LL)
                                         + *((_DWORD *)v51 + 82) * (unsigned int)((__int64)(v51[36] - v51[35]) >> 3)));
                  v63 = 0;
                  v62 = v39;
                  v40 = sub_CA1930(&v62);
                  v22 = v51;
                  v36 = v50;
                  if ( v59 < v40 )
                  {
LABEL_19:
                    v52 = v22;
                    *a6 = v49;
                    *a7 = v26;
                    v41 = *(unsigned int *)(v22[39]
                                          + 16LL
                                          * (*(unsigned __int16 *)(*(_QWORD *)v33 + 24LL)
                                           + *((_DWORD *)v22 + 82) * (unsigned int)((__int64)(v22[36] - v22[35]) >> 3)));
                    v63 = 0;
                    v62 = v41;
                    if ( sub_CA1930(&v62) == v36 )
                      return v33;
                    v55 = v33;
                    v22 = v52;
                  }
                }
              }
            }
          }
LABEL_21:
          v26 = *v25;
          v21 += v57;
          ++v25;
          if ( !v26 )
          {
            v19 = v27;
            v20 = v22;
            break;
          }
          v17 = v22[36];
          v23 = v22[35];
        }
      }
      v49 = (unsigned __int16)*v46;
      if ( !*v46 )
        break;
      v54 = (unsigned __int16)*v46;
      if ( a3 )
        v54 = (*(__int64 (__fastcall **)(_QWORD *))(*v20 + 296LL))(v20);
      ++v46;
      v17 = v20[36];
      v18 = v20[35];
      v19 += v44;
    }
  }
  return v55;
}
