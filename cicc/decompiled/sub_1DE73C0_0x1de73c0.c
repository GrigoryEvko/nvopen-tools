// Function: sub_1DE73C0
// Address: 0x1de73c0
//
__int64 __fastcall sub_1DE73C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  int v11; // edx
  __int64 *v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 v16; // r9
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rsi
  int v21; // edx
  int v22; // r8d
  unsigned int v23; // eax
  __int64 v24; // rdi
  int v25; // r15d
  unsigned __int64 v26; // r15
  __int64 v27; // rax
  unsigned int v28; // esi
  int v29; // r9d
  int v30; // r9d
  __int64 v31; // rsi
  int v32; // edx
  unsigned int v33; // r10d
  __int64 *v34; // rdi
  int v35; // edx
  int v36; // r11d
  int v37; // edx
  int v38; // r9d
  int v39; // r9d
  __int64 v40; // rsi
  int v41; // r15d
  __int64 *v42; // r11
  unsigned int v43; // r10d
  __int64 v44; // rbx
  int v45; // r11d
  __int64 *v46; // r15
  int v47; // [rsp+4h] [rbp-8Ch]
  __int64 v49; // [rsp+20h] [rbp-70h]
  __int64 *v52; // [rsp+38h] [rbp-58h]
  __int64 v53; // [rsp+38h] [rbp-58h]
  __int64 v54; // [rsp+40h] [rbp-50h] BYREF
  __int64 v55; // [rsp+48h] [rbp-48h] BYREF
  __int64 v56; // [rsp+50h] [rbp-40h] BYREF
  __int64 v57[7]; // [rsp+58h] [rbp-38h] BYREF

  result = 0;
  if ( *(_DWORD *)(a4 + 56) )
  {
    sub_15E44B0(**(_QWORD **)(a2 + 56));
    if ( v11 )
    {
      v12 = *(__int64 **)(a2 + 88);
      if ( (unsigned int)((__int64)(*(_QWORD *)(a2 + 96) - (_QWORD)v12) >> 3) == 2 )
      {
        v44 = v12[1];
        v53 = *v12;
        if ( sub_1DD6970(*v12, v44) || sub_1DD6970(v44, v53) )
        {
          sub_16AF710(v57, 2 * LODWORD(qword_4FC5840[20]), 0x96u);
          v47 = v57[0];
          goto LABEL_6;
        }
      }
      v13 = qword_4FC5840;
    }
    else
    {
      v13 = qword_4FC5920;
    }
    sub_16AF710(v57, *((_DWORD *)v13 + 40), 0x64u);
    v47 = v57[0];
LABEL_6:
    v57[0] = sub_20D7490(*(_QWORD *)(a1 + 568), a2);
    v14 = sub_16AF500(v57, a5);
    v15 = *(__int64 **)(a3 + 64);
    v54 = v14;
    v52 = *(__int64 **)(a3 + 72);
    if ( v52 == v15 )
      return 0;
    v49 = a1 + 888;
    while ( 1 )
    {
      v27 = *v15;
      v55 = v27;
      if ( a3 == v27 )
        goto LABEL_17;
      v28 = *(_DWORD *)(a1 + 912);
      if ( v28 )
      {
        v16 = *(_QWORD *)(a1 + 896);
        v17 = (v28 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v27 == *v18 )
        {
LABEL_9:
          if ( a4 == v18[1] )
            goto LABEL_17;
          goto LABEL_10;
        }
        v36 = 1;
        v34 = 0;
        while ( v19 != -8 )
        {
          if ( v19 == -16 && !v34 )
            v34 = v18;
          v17 = (v28 - 1) & (v36 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v27 == *v18 )
            goto LABEL_9;
          ++v36;
        }
        if ( !v34 )
          v34 = v18;
        v37 = *(_DWORD *)(a1 + 904);
        ++*(_QWORD *)(a1 + 888);
        v32 = v37 + 1;
        if ( 4 * v32 < 3 * v28 )
        {
          if ( v28 - *(_DWORD *)(a1 + 908) - v32 <= v28 >> 3 )
          {
            sub_1DE4DF0(v49, v28);
            v38 = *(_DWORD *)(a1 + 912);
            if ( !v38 )
            {
LABEL_70:
              ++*(_DWORD *)(a1 + 904);
              BUG();
            }
            v39 = v38 - 1;
            v40 = *(_QWORD *)(a1 + 896);
            v41 = 1;
            v42 = 0;
            v32 = *(_DWORD *)(a1 + 904) + 1;
            v43 = v39 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
            v34 = (__int64 *)(v40 + 16LL * v43);
            v27 = *v34;
            if ( v55 != *v34 )
            {
              while ( v27 != -8 )
              {
                if ( !v42 && v27 == -16 )
                  v42 = v34;
                v43 = v39 & (v41 + v43);
                v34 = (__int64 *)(v40 + 16LL * v43);
                v27 = *v34;
                if ( v55 == *v34 )
                  goto LABEL_23;
                ++v41;
              }
              v27 = v55;
              if ( v42 )
                v34 = v42;
            }
          }
          goto LABEL_23;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 888);
      }
      sub_1DE4DF0(v49, 2 * v28);
      v29 = *(_DWORD *)(a1 + 912);
      if ( !v29 )
        goto LABEL_70;
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 896);
      v32 = *(_DWORD *)(a1 + 904) + 1;
      v33 = v30 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
      v34 = (__int64 *)(v31 + 16LL * v33);
      v27 = *v34;
      if ( v55 != *v34 )
      {
        v45 = 1;
        v46 = 0;
        while ( v27 != -8 )
        {
          if ( !v46 && v27 == -16 )
            v46 = v34;
          v33 = v30 & (v45 + v33);
          v34 = (__int64 *)(v31 + 16LL * v33);
          v27 = *v34;
          if ( v55 == *v34 )
            goto LABEL_23;
          ++v45;
        }
        v27 = v55;
        if ( v46 )
          v34 = v46;
      }
LABEL_23:
      *(_DWORD *)(a1 + 904) = v32;
      if ( *v34 != -8 )
        --*(_DWORD *)(a1 + 908);
      *v34 = v27;
      v34[1] = 0;
LABEL_10:
      if ( !a7 )
        goto LABEL_14;
      if ( (*(_BYTE *)(a7 + 8) & 1) != 0 )
      {
        v20 = a7 + 16;
        v21 = 15;
      }
      else
      {
        v35 = *(_DWORD *)(a7 + 24);
        v20 = *(_QWORD *)(a7 + 16);
        if ( !v35 )
          goto LABEL_17;
        v21 = v35 - 1;
      }
      v22 = 1;
      v23 = v21 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
      v24 = *(_QWORD *)(v20 + 8LL * v23);
      if ( v55 != v24 )
      {
        while ( v24 != -8 )
        {
          v23 = v21 & (v22 + v23);
          v24 = *(_QWORD *)(v20 + 8LL * v23);
          if ( v55 == v24 )
            goto LABEL_14;
          ++v22;
        }
        if ( v52 == ++v15 )
          return 0;
      }
      else
      {
LABEL_14:
        if ( sub_1DE4FA0(v49, &v55)[1] != a6 && a2 != v55 )
        {
          v25 = sub_1DF1780(*(_QWORD *)(a1 + 560), v55, a3);
          v57[0] = sub_20D7490(*(_QWORD *)(a1 + 568), v55);
          v56 = sub_16AF500(v57, v25);
          v26 = sub_16AF500(&v54, 0x80000000 - v47);
          if ( v26 <= sub_16AF500(&v56, v47) )
            return 1;
        }
LABEL_17:
        if ( v52 == ++v15 )
          return 0;
      }
    }
  }
  return result;
}
