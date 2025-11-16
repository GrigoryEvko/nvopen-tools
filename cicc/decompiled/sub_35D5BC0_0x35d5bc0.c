// Function: sub_35D5BC0
// Address: 0x35d5bc0
//
__int64 __fastcall sub_35D5BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r8
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // rax
  _DWORD *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 (__fastcall *v22)(__int64, unsigned __int16); // rcx
  __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rsi
  unsigned int v26; // eax
  unsigned int v27; // esi
  unsigned int v28; // r15d
  __int64 v29; // r10
  unsigned __int64 *v30; // rdi
  int v31; // eax
  unsigned int v32; // r9d
  _QWORD *v33; // rdx
  unsigned __int64 v34; // r8
  unsigned int *v35; // rdx
  int v36; // r9d
  int v37; // eax
  int v38; // r8d
  unsigned int v39; // eax
  int v40; // edx
  int v41; // esi
  __int64 v42; // rcx
  unsigned int v43; // edx
  unsigned __int64 v44; // r9
  int v45; // r11d
  unsigned __int64 *v46; // r10
  int v47; // edx
  int v48; // edx
  __int64 v49; // r9
  int v50; // r11d
  unsigned int v51; // ecx
  unsigned __int64 v52; // rsi
  unsigned int v53; // [rsp+4h] [rbp-3Ch]
  __int64 (__fastcall *v54)(__int64, unsigned __int16); // [rsp+8h] [rbp-38h]
  __int64 v55; // [rsp+8h] [rbp-38h]

  v7 = a2 | 4;
  v8 = *(unsigned int *)(a1 + 120);
  v9 = *(_QWORD *)(a1 + 104);
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (v7 ^ (v7 >> 9));
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v7 == *v11 )
    {
LABEL_3:
      if ( v11 != (__int64 *)(v9 + 16 * v8) )
        return *((unsigned int *)v11 + 2);
    }
    else
    {
      v14 = 1;
      while ( v12 != -4 )
      {
        v36 = v14 + 1;
        v10 = (v8 - 1) & (v14 + v10);
        v11 = (__int64 *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( v7 == *v11 )
          goto LABEL_3;
        v14 = v36;
      }
    }
  }
  v15 = sub_2E79000(*(__int64 **)a1);
  v16 = *(_QWORD *)(a1 + 16);
  v17 = v15;
  v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v16 + 32LL);
  v54 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v16 + 552LL);
  if ( v18 == sub_2D42F30 )
  {
    v19 = sub_AE2980(v17, 0);
    v22 = v54;
    v23 = 2;
    v24 = v19[1];
    if ( v24 != 1 )
    {
      v23 = 3;
      if ( v24 != 2 )
      {
        v23 = 4;
        if ( v24 != 4 )
        {
          v23 = 5;
          if ( v24 != 8 )
          {
            v23 = 6;
            if ( v24 != 16 )
            {
              v23 = 7;
              if ( v24 != 32 )
              {
                v23 = 8;
                if ( v24 != 64 )
                  v23 = 9 * (unsigned int)(v24 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v39 = v18(v16, v17, 0);
    v22 = v54;
    v23 = v39;
  }
  if ( v22 == sub_2EC09E0 )
    v25 = *(_QWORD *)(v16 + 8LL * (unsigned __int16)v23 + 112);
  else
    v25 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v22)(v16, v23, 0);
  v55 = a1 + 96;
  v26 = sub_2EC06C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL), v25, byte_3F871B3, 0, v20, v21);
  v27 = *(_DWORD *)(a1 + 120);
  v28 = v26;
  if ( !v27 )
  {
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_40;
  }
  v29 = *(_QWORD *)(a1 + 104);
  v30 = 0;
  v31 = 1;
  v32 = (v27 - 1) & (v7 ^ (v7 >> 9));
  v33 = (_QWORD *)(v29 + 16LL * v32);
  v34 = *v33;
  if ( v7 != *v33 )
  {
    while ( v34 != -4 )
    {
      if ( v34 == -16 && !v30 )
        v30 = v33;
      v32 = (v27 - 1) & (v31 + v32);
      v33 = (_QWORD *)(v29 + 16LL * v32);
      v34 = *v33;
      if ( v7 == *v33 )
        goto LABEL_20;
      ++v31;
    }
    v37 = *(_DWORD *)(a1 + 112);
    if ( !v30 )
      v30 = v33;
    ++*(_QWORD *)(a1 + 96);
    v38 = v37 + 1;
    if ( 4 * (v37 + 1) < 3 * v27 )
    {
      if ( v27 - *(_DWORD *)(a1 + 116) - v38 > v27 >> 3 )
      {
LABEL_34:
        *(_DWORD *)(a1 + 112) = v38;
        if ( *v30 != -4 )
          --*(_DWORD *)(a1 + 116);
        *v30 = v7;
        v35 = (unsigned int *)(v30 + 1);
        *((_DWORD *)v30 + 2) = 0;
        goto LABEL_21;
      }
      v53 = v7 ^ (v7 >> 9);
      sub_35D59E0(v55, v27);
      v47 = *(_DWORD *)(a1 + 120);
      if ( v47 )
      {
        v48 = v47 - 1;
        v49 = *(_QWORD *)(a1 + 104);
        v46 = 0;
        v50 = 1;
        v51 = v48 & v53;
        v38 = *(_DWORD *)(a1 + 112) + 1;
        v30 = (unsigned __int64 *)(v49 + 16LL * (v48 & v53));
        v52 = *v30;
        if ( v7 == *v30 )
          goto LABEL_34;
        while ( v52 != -4 )
        {
          if ( !v46 && v52 == -16 )
            v46 = v30;
          v51 = v48 & (v50 + v51);
          v30 = (unsigned __int64 *)(v49 + 16LL * v51);
          v52 = *v30;
          if ( v7 == *v30 )
            goto LABEL_34;
          ++v50;
        }
        goto LABEL_44;
      }
      goto LABEL_60;
    }
LABEL_40:
    sub_35D59E0(v55, 2 * v27);
    v40 = *(_DWORD *)(a1 + 120);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 104);
      v38 = *(_DWORD *)(a1 + 112) + 1;
      v43 = (v40 - 1) & (v7 ^ (v7 >> 9));
      v30 = (unsigned __int64 *)(v42 + 16LL * v43);
      v44 = *v30;
      if ( v7 == *v30 )
        goto LABEL_34;
      v45 = 1;
      v46 = 0;
      while ( v44 != -4 )
      {
        if ( v44 == -16 && !v46 )
          v46 = v30;
        v43 = v41 & (v45 + v43);
        v30 = (unsigned __int64 *)(v42 + 16LL * v43);
        v44 = *v30;
        if ( v7 == *v30 )
          goto LABEL_34;
        ++v45;
      }
LABEL_44:
      if ( v46 )
        v30 = v46;
      goto LABEL_34;
    }
LABEL_60:
    ++*(_DWORD *)(a1 + 112);
    BUG();
  }
LABEL_20:
  v35 = (unsigned int *)(v33 + 1);
LABEL_21:
  *v35 = v28;
  sub_35D4CD0(a1, a3, a4, v28);
  return v28;
}
