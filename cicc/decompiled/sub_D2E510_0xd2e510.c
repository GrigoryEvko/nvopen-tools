// Function: sub_D2E510
// Address: 0xd2e510
//
__int64 *__fastcall sub_D2E510(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 *v10; // rdi
  __int64 *v11; // r13
  unsigned __int64 v12; // r14
  _QWORD *v13; // rcx
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  _QWORD *v16; // rax
  int v17; // r9d
  __int64 v18; // r11
  int v19; // r10d
  unsigned __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 *v22; // rcx
  __int64 v23; // r14
  _QWORD *v24; // rdx
  _QWORD *v25; // rcx
  __int64 *v26; // r13
  int v27; // r14d
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // r14
  __int64 *result; // rax
  __int64 v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // r12
  __int64 v40; // rax
  _QWORD *v41; // r14
  __int64 v42; // rdx
  int v43; // ecx
  int v44; // r8d
  __int64 v45; // rdx
  unsigned __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rsi
  __int64 v49; // r15
  __int64 v50; // r12
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  int v54; // r14d
  __int64 *v55; // rbx
  _QWORD *v56; // rax
  __int64 v57; // rdx
  __int64 *v59; // [rsp+28h] [rbp-88h]
  int v60; // [rsp+30h] [rbp-80h]
  char v61; // [rsp+40h] [rbp-70h]
  __int64 v62; // [rsp+40h] [rbp-70h]
  __int64 *v63; // [rsp+48h] [rbp-68h]
  _QWORD *v64; // [rsp+58h] [rbp-58h] BYREF
  __int64 *v65; // [rsp+60h] [rbp-50h] BYREF
  __int64 v66; // [rsp+68h] [rbp-48h]
  _QWORD v67[8]; // [rsp+70h] [rbp-40h] BYREF

  v5 = a1;
  v6 = sub_D29010(a1, a2);
  v7 = (__int64 *)sub_D23C40(a1, v6);
  v8 = (__int64)v7;
  if ( v7 )
    v8 = *v7;
  v61 = 0;
  v9 = v6 + 24;
  v10 = &a3[a4];
  v11 = a3;
  v63 = v10;
  if ( v10 == a3 )
    goto LABEL_44;
LABEL_4:
  while ( 2 )
  {
    while ( 1 )
    {
      v12 = sub_D29770(v5, *v11);
      sub_D25590(v9, v12, 0);
      v13 = *(_QWORD **)(v12 + 24);
      v14 = &v13[*(unsigned int *)(v12 + 32)];
      if ( v13 != v14 )
        break;
LABEL_19:
      if ( v63 == ++v11 )
        goto LABEL_20;
    }
    do
    {
      v15 = *v13;
      v16 = v13;
      if ( (*v13 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v13 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        if ( v13 == v14 )
          goto LABEL_19;
        v17 = *(_DWORD *)(v5 + 328);
        v18 = *(_QWORD *)(v5 + 312);
        v19 = v17 - 1;
        while ( 1 )
        {
          v20 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v17 )
            goto LABEL_34;
          v21 = v19 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v22 = (__int64 *)(v18 + 16LL * v21);
          v23 = *v22;
          if ( v20 != *v22 )
            break;
LABEL_13:
          v24 = (_QWORD *)v22[1];
          if ( v24 )
            v24 = (_QWORD *)*v24;
          if ( v24 == (_QWORD *)v8 )
          {
LABEL_35:
            v61 = 1;
            if ( v63 == ++v11 )
              goto LABEL_20;
            goto LABEL_4;
          }
LABEL_16:
          v25 = v16 + 1;
          if ( v14 != v16 + 1 )
          {
            while ( 1 )
            {
              v15 = *v25;
              v16 = v25;
              if ( (*v25 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                if ( *(_QWORD *)(*v25 & 0xFFFFFFFFFFFFFFF8LL) )
                  break;
              }
              if ( v14 == ++v25 )
                goto LABEL_19;
            }
            if ( v14 != v25 )
              continue;
          }
          goto LABEL_19;
        }
        v43 = 1;
        while ( v23 != -4096 )
        {
          v44 = v43 + 1;
          v21 = v19 & (v43 + v21);
          v22 = (__int64 *)(v18 + 16LL * v21);
          v23 = *v22;
          if ( v20 == *v22 )
            goto LABEL_13;
          v43 = v44;
        }
LABEL_34:
        if ( !v8 )
          goto LABEL_35;
        goto LABEL_16;
      }
      ++v13;
    }
    while ( v14 != v13 );
    if ( v63 != ++v11 )
      continue;
    break;
  }
LABEL_20:
  if ( v61 )
  {
LABEL_21:
    v26 = a3;
    do
    {
      v34 = *v26;
      v35 = sub_D29010(v5, *v26);
      *(_QWORD *)(v5 + 288) += 32LL;
      v39 = (__int64 *)v35;
      v67[0] = v35;
      v65 = v67;
      v66 = 0x100000001LL;
      v40 = *(_QWORD *)(v5 + 208);
      v41 = (_QWORD *)((v40 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      v42 = (__int64)(v41 + 4);
      if ( *(_QWORD *)(v5 + 216) >= (unsigned __int64)(v41 + 4) && v40 )
      {
        *(_QWORD *)(v5 + 208) = v42;
      }
      else
      {
        v34 = 32;
        v41 = (_QWORD *)sub_9D1E70(v5 + 208, 32, 32, 3);
      }
      *v41 = v8;
      v41[1] = v41 + 3;
      v41[2] = 0x100000000LL;
      if ( (_DWORD)v66 )
      {
        v34 = (__int64)&v65;
        sub_D230A0((__int64)(v41 + 1), (char **)&v65, v42, v36, v37, v38);
      }
      v64 = v41;
      if ( v65 != v67 )
        _libc_free(v65, v34);
      v27 = *(_DWORD *)(v8 + 64) >> 1;
      *(_DWORD *)sub_D25AF0(v8 + 56, (__int64 *)&v64) = v27;
      v30 = *(unsigned int *)(v8 + 16);
      v31 = (__int64)v64;
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 20) )
      {
        sub_C8D5F0(v8 + 8, (const void *)(v8 + 24), v30 + 1, 8u, v28, v29);
        v30 = *(unsigned int *)(v8 + 16);
      }
      ++v26;
      *(_QWORD *)(*(_QWORD *)(v8 + 8) + 8 * v30) = v31;
      v32 = (__int64)v64;
      ++*(_DWORD *)(v8 + 16);
      v65 = v39;
      result = sub_D25E90(v5 + 304, (__int64 *)&v65);
      *result = v32;
    }
    while ( v63 != v26 );
    return result;
  }
LABEL_44:
  v45 = *(_QWORD *)(v5 + 336);
  *(_QWORD *)(v5 + 416) += 136LL;
  v46 = ((v45 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 136;
  if ( *(_QWORD *)(v5 + 344) >= v46 && v45 )
  {
    *(_QWORD *)(v5 + 336) = v46;
    v59 = (__int64 *)((v45 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    v59 = (__int64 *)sub_9D1E70(v5 + 336, 136, 136, 3);
  }
  v47 = v5 + 576;
  sub_D23F30(v59, v5);
  sub_D248B0(&v65, (__int64 *)(v5 + 576), v8);
  v48 = *(_QWORD *)(v5 + 432);
  v49 = *(int *)(v67[0] + 8LL);
  v50 = 8 * v49;
  v65 = v59;
  sub_D23810(v5 + 432, (char *)(8 * v49 + v48), (__int64 *)&v65, v51, v52, v53);
  result = (__int64 *)*(unsigned int *)(v5 + 440);
  v60 = (int)result;
  if ( (int)v49 < (int)result )
  {
    v54 = v49;
    v62 = v5;
    do
    {
      v55 = (__int64 *)(v50 + *(_QWORD *)(v62 + 432));
      if ( (unsigned __int8)sub_D24D10(v47, v55, &v65) )
      {
        result = v65 + 1;
      }
      else
      {
        v56 = sub_D27750(v47, v55, v65);
        v57 = *v55;
        result = v56 + 1;
        *(_DWORD *)result = 0;
        *(result - 1) = v57;
      }
      *(_DWORD *)result = v54;
      v50 += 8;
      ++v54;
    }
    while ( v60 != v54 );
    v5 = v62;
  }
  if ( v63 != a3 )
  {
    v8 = (__int64)v59;
    goto LABEL_21;
  }
  return result;
}
