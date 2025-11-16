// Function: sub_386A280
// Address: 0x386a280
//
__int64 __fastcall sub_386A280(__int64 a1, __int64 *a2, __m128i a3, __m128i a4)
{
  __int64 v5; // rbx
  unsigned int v6; // eax
  unsigned int v7; // r12d
  __int64 v9; // r14
  __int16 v10; // ax
  __int64 v11; // r13
  unsigned int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // r15
  __int64 v18; // rdi
  int v19; // r10d
  __int64 *v20; // r9
  int v21; // ecx
  int v22; // ecx
  int v23; // eax
  int v24; // esi
  __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 v27; // r8
  int v28; // r10d
  __int64 *v29; // r9
  int v30; // eax
  int v31; // edx
  __int64 v32; // rdi
  __int64 *v33; // r8
  unsigned int v34; // r15d
  int v35; // r9d
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rax
  unsigned int v40; // esi
  __int64 v41; // r12
  __int64 v42; // r14
  __int64 v43; // r8
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // rdi
  int v47; // ecx
  __int64 *v48; // r10
  int v49; // ecx
  int v50; // edx
  __int64 v51; // [rsp+0h] [rbp-40h] BYREF
  __int64 v52[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = (__int64)a2;
  LOBYTE(v6) = sub_1456C80(*(_QWORD *)(a1 + 48), *a2);
  if ( !(_BYTE)v6 )
    return 0;
  v7 = v6;
  v9 = sub_146F1B0(*(_QWORD *)(a1 + 48), (__int64)a2);
  v10 = *(_WORD *)(v9 + 24);
  if ( !v10 )
  {
    v11 = *(_QWORD *)(a1 + 40);
    v12 = *(_DWORD *)(v11 + 24);
    if ( v12 )
    {
      v13 = *(_QWORD *)(v11 + 8);
      v14 = (v12 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( v5 == *v15 )
      {
LABEL_7:
        v15[1] = *(_QWORD *)(v9 + 32);
        return v7;
      }
      v19 = 1;
      v20 = 0;
      while ( v16 != -8 )
      {
        if ( !v20 && v16 == -16 )
          v20 = v15;
        v14 = (v12 - 1) & (v19 + v14);
        v15 = (__int64 *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( v5 == *v15 )
          goto LABEL_7;
        ++v19;
      }
      v21 = *(_DWORD *)(v11 + 16);
      if ( v20 )
        v15 = v20;
      ++*(_QWORD *)v11;
      v22 = v21 + 1;
      if ( 4 * v22 < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(v11 + 20) - v22 > v12 >> 3 )
        {
LABEL_18:
          *(_DWORD *)(v11 + 16) = v22;
          if ( *v15 != -8 )
            --*(_DWORD *)(v11 + 20);
          *v15 = v5;
          v15[1] = 0;
          goto LABEL_7;
        }
        sub_19B8820(v11, v12);
        v30 = *(_DWORD *)(v11 + 24);
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *(_QWORD *)(v11 + 8);
          v33 = 0;
          v34 = (v30 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v35 = 1;
          v22 = *(_DWORD *)(v11 + 16) + 1;
          v15 = (__int64 *)(v32 + 16LL * v34);
          v36 = *v15;
          if ( v5 != *v15 )
          {
            while ( v36 != -8 )
            {
              if ( !v33 && v36 == -16 )
                v33 = v15;
              v34 = v31 & (v35 + v34);
              v15 = (__int64 *)(v32 + 16LL * v34);
              v36 = *v15;
              if ( v5 == *v15 )
                goto LABEL_18;
              ++v35;
            }
            if ( v33 )
              v15 = v33;
          }
          goto LABEL_18;
        }
LABEL_72:
        ++*(_DWORD *)(v11 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v11;
    }
    sub_19B8820(v11, 2 * v12);
    v23 = *(_DWORD *)(v11 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v11 + 8);
      v26 = (v23 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v22 = *(_DWORD *)(v11 + 16) + 1;
      v15 = (__int64 *)(v25 + 16LL * v26);
      v27 = *v15;
      if ( v5 != *v15 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -8 )
        {
          if ( !v29 && v27 == -16 )
            v29 = v15;
          v26 = v24 & (v28 + v26);
          v15 = (__int64 *)(v25 + 16LL * v26);
          v27 = *v15;
          if ( v5 == *v15 )
            goto LABEL_18;
          ++v28;
        }
        if ( v29 )
          v15 = v29;
      }
      goto LABEL_18;
    }
    goto LABEL_72;
  }
  if ( v10 != 7 || *(_QWORD *)(v9 + 48) != *(_QWORD *)(a1 + 56) )
    return 0;
  v17 = sub_1487810(v9, *(_QWORD *)(a1 + 32), *(_QWORD **)(a1 + 48), a3, a4);
  if ( !*((_WORD *)v17 + 12) )
  {
    v18 = *(_QWORD *)(a1 + 40);
    v52[0] = (__int64)a2;
    sub_38526A0(v18, v52)[1] = v17[4];
    return v7;
  }
  v37 = sub_1456F20(*(_QWORD *)(a1 + 48), v9);
  v38 = v37;
  if ( *(_WORD *)(v37 + 24) != 10 )
    return 0;
  v39 = sub_14806B0(*(_QWORD *)(a1 + 48), (__int64)v17, v37, 0, 0);
  if ( *(_WORD *)(v39 + 24) )
    return 0;
  v40 = *(_DWORD *)(a1 + 24);
  v41 = *(_QWORD *)(v38 - 8);
  v51 = v5;
  v42 = *(_QWORD *)(v39 + 32);
  if ( !v40 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_55;
  }
  v43 = *(_QWORD *)(a1 + 8);
  v44 = (v40 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v45 = (__int64 *)(v43 + 24LL * v44);
  v46 = *v45;
  if ( v5 != *v45 )
  {
    v47 = 1;
    v48 = 0;
    while ( v46 != -8 )
    {
      if ( v46 == -16 && !v48 )
        v48 = v45;
      v44 = (v40 - 1) & (v47 + v44);
      v45 = (__int64 *)(v43 + 24LL * v44);
      v46 = *v45;
      if ( v5 == *v45 )
        goto LABEL_39;
      ++v47;
    }
    v49 = *(_DWORD *)(a1 + 16);
    if ( v48 )
      v45 = v48;
    ++*(_QWORD *)a1;
    v50 = v49 + 1;
    if ( 4 * (v49 + 1) < 3 * v40 )
    {
      if ( v40 - *(_DWORD *)(a1 + 20) - v50 > v40 >> 3 )
      {
LABEL_51:
        *(_DWORD *)(a1 + 16) = v50;
        if ( *v45 != -8 )
          --*(_DWORD *)(a1 + 20);
        *v45 = v5;
        v45[1] = 0;
        v45[2] = 0;
        goto LABEL_39;
      }
LABEL_56:
      sub_386A0B0(a1, v40);
      sub_3869E30(a1, &v51, v52);
      v45 = (__int64 *)v52[0];
      v5 = v51;
      v50 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_51;
    }
LABEL_55:
    v40 *= 2;
    goto LABEL_56;
  }
LABEL_39:
  v45[1] = v41;
  v45[2] = v42;
  return 0;
}
