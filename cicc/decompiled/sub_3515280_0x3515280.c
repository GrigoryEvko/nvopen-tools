// Function: sub_3515280
// Address: 0x3515280
//
unsigned __int64 __fastcall sub_3515280(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 *v5; // r13
  _QWORD *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // rsi
  int v11; // edx
  __int64 v12; // rcx
  int v13; // edx
  unsigned int v14; // eax
  __int64 v15; // rdi
  int v16; // r9d
  unsigned int v17; // eax
  __int64 v18; // rsi
  __int64 *v19; // r12
  _QWORD *v20; // rdi
  _QWORD *v21; // rsi
  _QWORD **v22; // r8
  __int64 v23; // rbx
  __int64 v24; // rdi
  unsigned int v25; // eax
  unsigned int v26; // esi
  unsigned int v27; // r13d
  __int64 v28; // r9
  int v29; // r11d
  __int64 *v30; // rcx
  unsigned int v31; // r8d
  __int64 *v32; // rax
  __int64 v33; // rdi
  int v34; // edx
  __int64 v35; // rcx
  int v36; // edx
  unsigned int v37; // eax
  __int64 v38; // rsi
  int v39; // edi
  int v40; // eax
  int v41; // eax
  int v43; // r9d
  int v44; // r9d
  __int64 v45; // rdi
  unsigned int v46; // edx
  __int64 v47; // rsi
  int v48; // r11d
  __int64 *v49; // r10
  int v50; // esi
  int v51; // esi
  __int64 v52; // rdi
  int v53; // r11d
  __int64 v54; // r9
  __int64 v55; // rdx
  unsigned int v56; // r12d
  unsigned __int64 v57; // rax
  __int64 *v58; // [rsp+8h] [rbp-88h]
  unsigned __int64 v60; // [rsp+18h] [rbp-78h]
  __int64 *v61; // [rsp+28h] [rbp-68h]
  unsigned int v62; // [rsp+34h] [rbp-5Ch]
  __int64 v63; // [rsp+38h] [rbp-58h]
  __int64 *v64; // [rsp+48h] [rbp-48h]
  __int64 v65; // [rsp+50h] [rbp-40h] BYREF
  __int64 v66[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = *(__int64 **)(a2 + 64);
  v61 = &v4[*(unsigned int *)(a2 + 72)];
  if ( v61 != v4 )
  {
    v5 = *(__int64 **)(a2 + 64);
    v63 = a1 + 888;
    v60 = 0;
    while ( 1 )
    {
      v65 = *v5;
      v9 = *sub_3515040(v63, &v65);
      if ( *(_DWORD *)(a3 + 16) )
      {
        v11 = *(_DWORD *)(a3 + 24);
        v10 = v65;
        v12 = *(_QWORD *)(a3 + 8);
        if ( v11 )
        {
          v13 = v11 - 1;
          v14 = v13 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
          v15 = *(_QWORD *)(v12 + 8LL * v14);
          if ( v65 == v15 )
            goto LABEL_7;
          v16 = 1;
          while ( v15 != -4096 )
          {
            v14 = v13 & (v16 + v14);
            v15 = *(_QWORD *)(v12 + 8LL * v14);
            if ( v65 == v15 )
              goto LABEL_7;
            ++v16;
          }
        }
      }
      else
      {
        v7 = *(_QWORD **)(a3 + 32);
        v8 = &v7[*(unsigned int *)(a3 + 40)];
        if ( v8 != sub_3510810(v7, (__int64)v8, &v65) )
          goto LABEL_7;
        v10 = v65;
      }
      if ( v9 && *(_QWORD *)(*(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 8) - 8) != v10 )
        goto LABEL_7;
      v17 = sub_2E441D0(*(_QWORD *)(a1 + 528), v10, a2);
      v18 = v65;
      v62 = v17;
      v19 = *(__int64 **)(v65 + 112);
      v64 = &v19[*(unsigned int *)(v65 + 120)];
      if ( v19 != v64 )
        break;
LABEL_64:
      v56 = sub_2E441D0(*(_QWORD *)(a1 + 528), v18, a2);
      v66[0] = sub_2F06CB0(*(_QWORD *)(a1 + 536), v65);
      v57 = sub_1098D20((unsigned __int64 *)v66, v56);
      if ( v60 >= v57 )
        v57 = v60;
      v60 = v57;
LABEL_7:
      if ( v61 == ++v5 )
        return v60;
    }
    v58 = v5;
    while ( 1 )
    {
      v23 = *v19;
      v24 = *(_QWORD *)(a1 + 528);
      v66[0] = *v19;
      v25 = sub_2E441D0(v24, v18, v66[0]);
      v26 = *(_DWORD *)(a1 + 912);
      v27 = v25;
      if ( v26 )
      {
        v28 = *(_QWORD *)(a1 + 896);
        v29 = 1;
        v30 = 0;
        v31 = (v26 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v32 = (__int64 *)(v28 + 16LL * v31);
        v33 = *v32;
        if ( v23 == *v32 )
        {
LABEL_24:
          v22 = (_QWORD **)v32[1];
          goto LABEL_25;
        }
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v30 )
            v30 = v32;
          v31 = (v26 - 1) & (v29 + v31);
          v32 = (__int64 *)(v28 + 16LL * v31);
          v33 = *v32;
          if ( v23 == *v32 )
            goto LABEL_24;
          ++v29;
        }
        if ( !v30 )
          v30 = v32;
        v40 = *(_DWORD *)(a1 + 904);
        ++*(_QWORD *)(a1 + 888);
        v41 = v40 + 1;
        if ( 4 * v41 < 3 * v26 )
        {
          if ( v26 - *(_DWORD *)(a1 + 908) - v41 > v26 >> 3 )
            goto LABEL_42;
          sub_3512300(v63, v26);
          v50 = *(_DWORD *)(a1 + 912);
          if ( !v50 )
          {
LABEL_73:
            ++*(_DWORD *)(a1 + 904);
            BUG();
          }
          v51 = v50 - 1;
          v52 = *(_QWORD *)(a1 + 896);
          v49 = 0;
          v53 = 1;
          v41 = *(_DWORD *)(a1 + 904) + 1;
          v54 = v51 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v30 = (__int64 *)(v52 + 16 * v54);
          v55 = *v30;
          if ( v23 == *v30 )
            goto LABEL_42;
          while ( v55 != -4096 )
          {
            if ( !v49 && v55 == -8192 )
              v49 = v30;
            LODWORD(v54) = v51 & (v53 + v54);
            v30 = (__int64 *)(v52 + 16LL * (unsigned int)v54);
            v55 = *v30;
            if ( v23 == *v30 )
              goto LABEL_42;
            ++v53;
          }
          goto LABEL_52;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 888);
      }
      sub_3512300(v63, 2 * v26);
      v43 = *(_DWORD *)(a1 + 912);
      if ( !v43 )
        goto LABEL_73;
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 896);
      v46 = v44 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v41 = *(_DWORD *)(a1 + 904) + 1;
      v30 = (__int64 *)(v45 + 16LL * v46);
      v47 = *v30;
      if ( v23 == *v30 )
        goto LABEL_42;
      v48 = 1;
      v49 = 0;
      while ( v47 != -4096 )
      {
        if ( !v49 && v47 == -8192 )
          v49 = v30;
        v46 = v44 & (v48 + v46);
        v30 = (__int64 *)(v45 + 16LL * v46);
        v47 = *v30;
        if ( v23 == *v30 )
          goto LABEL_42;
        ++v48;
      }
LABEL_52:
      if ( v49 )
        v30 = v49;
LABEL_42:
      *(_DWORD *)(a1 + 904) = v41;
      if ( *v30 != -4096 )
        --*(_DWORD *)(a1 + 908);
      *v30 = v23;
      v22 = 0;
      v30[1] = 0;
LABEL_25:
      if ( *(_DWORD *)(a3 + 16) )
      {
        v34 = *(_DWORD *)(a3 + 24);
        v35 = *(_QWORD *)(a3 + 8);
        if ( v34 )
        {
          v36 = v34 - 1;
          v37 = v36 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v38 = *(_QWORD *)(v35 + 8LL * v37);
          if ( v23 == v38 )
            goto LABEL_21;
          v39 = 1;
          while ( v38 != -4096 )
          {
            v37 = v36 & (v39 + v37);
            v38 = *(_QWORD *)(v35 + 8LL * v37);
            if ( v23 == v38 )
              goto LABEL_21;
            ++v39;
          }
        }
      }
      else
      {
        v20 = *(_QWORD **)(a3 + 32);
        v21 = &v20[*(unsigned int *)(a3 + 40)];
        if ( v21 != sub_3510810(v20, (__int64)v21, v66) )
          goto LABEL_21;
      }
      if ( v27 > v62 && (!v22 || v23 == **v22) )
      {
        v5 = v58;
        goto LABEL_7;
      }
LABEL_21:
      v18 = v65;
      if ( v64 == ++v19 )
      {
        v5 = v58;
        goto LABEL_64;
      }
    }
  }
  return 0;
}
