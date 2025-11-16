// Function: sub_32CA770
// Address: 0x32ca770
//
__int64 __fastcall sub_32CA770(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rsi
  int v11; // r15d
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 v15; // rdi
  __int64 (*v16)(); // r8
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 *v21; // r14
  __int64 *v22; // r13
  __int64 v23; // r12
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r11
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r12
  unsigned __int16 *v33; // rax
  __int64 v34; // r13
  unsigned int v35; // r14d
  __int64 v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // r9
  __int64 v40; // r12
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // r14
  __int64 v44; // r8
  __int64 *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r10
  __int64 v49; // r11
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned __int16 *v52; // rax
  __int64 v53; // r14
  int v54; // r12d
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r10
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int128 v65; // [rsp-10h] [rbp-F0h]
  __int128 v66; // [rsp-10h] [rbp-F0h]
  __int64 v67; // [rsp+0h] [rbp-E0h]
  __int64 v68; // [rsp+0h] [rbp-E0h]
  __int64 v69; // [rsp+0h] [rbp-E0h]
  __int64 v70; // [rsp+0h] [rbp-E0h]
  __int64 v71; // [rsp+8h] [rbp-D8h]
  __int64 v72; // [rsp+18h] [rbp-C8h]
  __int64 v73; // [rsp+18h] [rbp-C8h]
  __int64 v74; // [rsp+18h] [rbp-C8h]
  __int64 v75; // [rsp+18h] [rbp-C8h]
  __int64 v76; // [rsp+20h] [rbp-C0h]
  __int64 v77; // [rsp+28h] [rbp-B8h]
  __int128 v78; // [rsp+30h] [rbp-B0h]
  __int64 v79; // [rsp+30h] [rbp-B0h]
  __int64 v80; // [rsp+48h] [rbp-98h] BYREF
  __int64 v81; // [rsp+50h] [rbp-90h] BYREF
  int v82; // [rsp+58h] [rbp-88h]
  __int128 v83; // [rsp+60h] [rbp-80h] BYREF
  _BYTE v84[112]; // [rsp+70h] [rbp-70h] BYREF

  *(_QWORD *)&v78 = a2;
  v10 = *(_QWORD *)(a6 + 80);
  *((_QWORD *)&v78 + 1) = a3;
  v81 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v81, v10, 1);
  v82 = *(_DWORD *)(a6 + 72);
  v11 = **(unsigned __int16 **)(a6 + 48);
  v77 = *(_QWORD *)(*(_QWORD *)(a6 + 48) + 8LL);
  if ( (unsigned __int8)sub_326A930(a4, a5, 1u) )
  {
    v27 = sub_3289780(a1, a4, a5, (__int64)&v81, 0, 0, v83, 0);
    v72 = v28;
    v29 = v27;
    if ( v27 )
    {
      if ( *(_DWORD *)(v27 + 24) != 328 )
      {
        *(_QWORD *)&v83 = v27;
        v67 = v27;
        sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v83);
        v29 = v67;
        if ( *(int *)(v67 + 88) < 0 )
        {
          *(_DWORD *)(v67 + 88) = *((_DWORD *)a1 + 12);
          v62 = *((unsigned int *)a1 + 12);
          if ( v62 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
          {
            sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v62 + 1, 8u, v30, v31);
            v62 = *((unsigned int *)a1 + 12);
            v29 = v67;
          }
          *(_QWORD *)(a1[5] + 8 * v62) = v29;
          ++*((_DWORD *)a1 + 12);
        }
      }
      v68 = v29;
      v32 = a1[1];
      v33 = (unsigned __int16 *)(*(_QWORD *)(v78 + 48) + 16LL * DWORD2(v78));
      v34 = *((_QWORD *)v33 + 1);
      v35 = *v33;
      v36 = sub_2E79000(*(__int64 **)(*a1 + 40));
      v37 = sub_2FE6750(v32, v35, v34, v36);
      v40 = sub_33FB310(*a1, v68, v72, &v81, v37, v38);
      v42 = v41;
      v43 = v40;
      if ( *(_DWORD *)(v40 + 24) == 328 )
        goto LABEL_26;
      goto LABEL_25;
    }
  }
  if ( *(_DWORD *)(a4 + 24) == 190 )
  {
    v45 = *(__int64 **)(a4 + 40);
    v73 = *v45;
    v76 = v45[1];
    if ( (unsigned __int8)sub_326A930(*v45, v76, 1u) )
    {
      v46 = sub_3289780(a1, v73, v76, (__int64)&v81, 0, 0, v83, 0);
      v48 = v46;
      v49 = v47;
      if ( v46 )
      {
        if ( *(_DWORD *)(v46 + 24) != 328 )
        {
          *(_QWORD *)&v83 = v46;
          v69 = v47;
          v74 = v46;
          sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v83);
          v48 = v74;
          v49 = v69;
          if ( *(int *)(v74 + 88) < 0 )
          {
            *(_DWORD *)(v74 + 88) = *((_DWORD *)a1 + 12);
            v63 = *((unsigned int *)a1 + 12);
            if ( v63 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
            {
              sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v63 + 1, 8u, v50, v51);
              v63 = *((unsigned int *)a1 + 12);
              v49 = v69;
              v48 = v74;
            }
            *(_QWORD *)(a1[5] + 8 * v63) = v48;
            ++*((_DWORD *)a1 + 12);
          }
        }
        v52 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 40LL) + 48LL)
                                 + 16LL * *(unsigned int *)(*(_QWORD *)(a4 + 40) + 48LL));
        v53 = *((_QWORD *)v52 + 1);
        v54 = *v52;
        v55 = sub_33FB310(*a1, v48, v49, &v81, *v52, v53);
        v75 = v55;
        v57 = v55;
        v58 = v56;
        if ( *(_DWORD *)(v55 + 24) != 328 )
        {
          *(_QWORD *)&v83 = v55;
          v70 = v55;
          v71 = v56;
          sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v83);
          v59 = v75;
          v57 = v70;
          v58 = v71;
          if ( *(int *)(v75 + 88) < 0 )
          {
            *(_DWORD *)(v75 + 88) = *((_DWORD *)a1 + 12);
            v64 = *((unsigned int *)a1 + 12);
            if ( v64 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
            {
              sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v64 + 1, 8u, v70, v71);
              v64 = *((unsigned int *)a1 + 12);
              v57 = v70;
              v58 = v71;
              v59 = v75;
            }
            *(_QWORD *)(a1[5] + 8 * v64) = v59;
            ++*((_DWORD *)a1 + 12);
          }
        }
        *((_QWORD *)&v66 + 1) = v58;
        *(_QWORD *)&v66 = v57;
        v40 = sub_3406EB0(*a1, 56, (unsigned int)&v81, v54, v53, v58, *(_OWORD *)(*(_QWORD *)(a4 + 40) + 40LL), v66);
        v42 = v60;
        v43 = v40;
        if ( *(_DWORD *)(v40 + 24) == 328 )
          goto LABEL_26;
LABEL_25:
        *(_QWORD *)&v83 = v40;
        sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v83);
        if ( *(int *)(v43 + 88) < 0 )
        {
          *(_DWORD *)(v43 + 88) = *((_DWORD *)a1 + 12);
          v61 = *((unsigned int *)a1 + 12);
          if ( v61 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
          {
            sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v61 + 1, 8u, v44, v39);
            v61 = *((unsigned int *)a1 + 12);
          }
          *(_QWORD *)(a1[5] + 8 * v61) = v43;
          ++*((_DWORD *)a1 + 12);
        }
LABEL_26:
        *((_QWORD *)&v65 + 1) = v42;
        *(_QWORD *)&v65 = v40;
        v13 = sub_3406EB0(*a1, 192, (unsigned int)&v81, v11, v77, v39, v78, v65);
        goto LABEL_7;
      }
    }
  }
  v12 = *(_QWORD *)(**(_QWORD **)(*a1 + 40) + 120LL);
  if ( !(unsigned __int8)sub_326A930(a4, a5, 0)
    || (v15 = a1[1], v16 = *(__int64 (**)())(*(_QWORD *)v15 + 200LL), v16 != sub_2FE2F30)
    && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v16)(
         v15,
         **(unsigned __int16 **)(a6 + 48),
         *(_QWORD *)(*(_QWORD *)(a6 + 48) + 8LL),
         v12)
    || (unsigned __int8)sub_B2D610(**(_QWORD **)(*a1 + 40), 18) )
  {
LABEL_6:
    v13 = 0;
    goto LABEL_7;
  }
  v17 = *((unsigned __int8 *)a1 + 33);
  v18 = a1[1];
  v19 = *((unsigned __int8 *)a1 + 34);
  v20 = *a1;
  *(_QWORD *)&v83 = v84;
  *((_QWORD *)&v83 + 1) = 0x800000000LL;
  v79 = sub_3489D70(v18, a6, v20, v17, v19, &v83);
  if ( !v79 )
  {
    if ( (_BYTE *)v83 != v84 )
      _libc_free(v83);
    goto LABEL_6;
  }
  v21 = (__int64 *)(v83 + 8LL * DWORD2(v83));
  if ( (__int64 *)v83 != v21 )
  {
    v22 = (__int64 *)v83;
    do
    {
      v23 = *v22;
      if ( *(_DWORD *)(*v22 + 24) != 328 )
      {
        v80 = *v22;
        sub_32B3B20((__int64)(a1 + 71), &v80);
        if ( *(int *)(v23 + 88) < 0 )
        {
          *(_DWORD *)(v23 + 88) = *((_DWORD *)a1 + 12);
          v26 = *((unsigned int *)a1 + 12);
          if ( v26 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
          {
            sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v26 + 1, 8u, v24, v25);
            v26 = *((unsigned int *)a1 + 12);
          }
          *(_QWORD *)(a1[5] + 8 * v26) = v23;
          ++*((_DWORD *)a1 + 12);
        }
      }
      ++v22;
    }
    while ( v21 != v22 );
    v21 = (__int64 *)v83;
  }
  if ( v21 != (__int64 *)v84 )
    _libc_free((unsigned __int64)v21);
  v13 = v79;
LABEL_7:
  if ( v81 )
    sub_B91220((__int64)&v81, v81);
  return v13;
}
