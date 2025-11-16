// Function: sub_D0CAF0
// Address: 0xd0caf0
//
__int64 __fastcall sub_D0CAF0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, void *a5, __int64 a6)
{
  __int64 v10; // r13
  char v11; // si
  __int64 v12; // rdi
  int v13; // edx
  unsigned int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // r13
  char v19; // si
  __int64 v20; // rcx
  int v21; // edi
  __int64 v22; // rax
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  int v30; // r10d
  unsigned int i; // eax
  __int64 v32; // rsi
  unsigned int v33; // eax
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned int j; // eax
  __int64 v40; // rsi
  int v41; // eax
  __int64 v42; // rdi
  __int64 (*v43)(); // r9
  char v44; // al
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 (__fastcall *v47)(__int64, __int64, __int64, __int64); // rax
  char v48; // al
  int v49; // eax
  int v50; // r10d
  void *v51; // [rsp+10h] [rbp-70h] BYREF
  char v52[8]; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v53[16]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v54; // [rsp+30h] [rbp-50h]

  v10 = *a4;
  v11 = *(_BYTE *)(*a4 + 8) & 1;
  if ( v11 )
  {
    v12 = v10 + 16;
    v13 = 7;
  }
  else
  {
    v26 = *(unsigned int *)(v10 + 24);
    v12 = *(_QWORD *)(v10 + 16);
    if ( !(_DWORD)v26 )
      goto LABEL_27;
    v13 = v26 - 1;
  }
  a5 = &unk_4F86630;
  v14 = v13 & (((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4));
  v15 = v12 + 16LL * v14;
  a6 = *(_QWORD *)v15;
  if ( *(_UNKNOWN **)v15 == &unk_4F86630 )
    goto LABEL_4;
  v35 = 1;
  while ( a6 != -4096 )
  {
    v50 = v35 + 1;
    v14 = v13 & (v35 + v14);
    v15 = v12 + 16LL * v14;
    a6 = *(_QWORD *)v15;
    if ( *(_UNKNOWN **)v15 == &unk_4F86630 )
      goto LABEL_4;
    v35 = v50;
  }
  if ( v11 )
  {
    v34 = 128;
    goto LABEL_28;
  }
  v26 = *(unsigned int *)(v10 + 24);
LABEL_27:
  v34 = 16 * v26;
LABEL_28:
  v15 = v12 + v34;
LABEL_4:
  v16 = 128;
  if ( !v11 )
    v16 = 16LL * *(unsigned int *)(v10 + 24);
  if ( v15 == v12 + v16 )
  {
    v27 = a4[1];
    v28 = *(unsigned int *)(v27 + 24);
    v29 = *(_QWORD *)(v27 + 8);
    if ( (_DWORD)v28 )
    {
      v30 = 1;
      for ( i = (v28 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v28 - 1) & v33 )
      {
        v32 = v29 + 24LL * i;
        if ( *(_UNKNOWN **)v32 == &unk_4F86630 && a2 == *(_QWORD *)(v32 + 8) )
          break;
        if ( *(_QWORD *)v32 == -4096 && *(_QWORD *)(v32 + 8) == -4096 )
          goto LABEL_39;
        v33 = v30 + i;
        ++v30;
      }
    }
    else
    {
LABEL_39:
      v32 = v29 + 24 * v28;
    }
    v42 = *(_QWORD *)(*(_QWORD *)(v32 + 16) + 24LL);
    v43 = *(__int64 (**)())(*(_QWORD *)v42 + 16LL);
    v44 = 0;
    if ( v43 != sub_D000F0 )
      v44 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v43)(v42, a2, a3, a4);
    v52[0] = v44;
    v51 = &unk_4F86630;
    sub_BBCF50((__int64)v53, v10, (__int64 *)&v51, v52);
    v15 = v54;
  }
  result = *(unsigned __int8 *)(v15 + 8);
  if ( !(_BYTE)result && *(_QWORD *)(a1 + 32) )
  {
    v18 = *a4;
    v19 = *(_BYTE *)(*a4 + 8) & 1;
    if ( v19 )
    {
      v20 = v18 + 16;
      v21 = 7;
    }
    else
    {
      v22 = *(unsigned int *)(v18 + 24);
      v20 = *(_QWORD *)(v18 + 16);
      if ( !(_DWORD)v22 )
        goto LABEL_44;
      v21 = v22 - 1;
    }
    v23 = v21 & (((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4));
    v24 = v20 + 16LL * v23;
    a5 = *(void **)v24;
    if ( *(_UNKNOWN **)v24 == &unk_4F81450 )
    {
LABEL_14:
      v25 = 128;
      if ( !v19 )
        v25 = 16LL * *(unsigned int *)(v18 + 24);
      if ( v24 == v20 + v25 )
      {
        v36 = a4[1];
        v37 = *(unsigned int *)(v36 + 24);
        v38 = *(_QWORD *)(v36 + 8);
        if ( (_DWORD)v37 )
        {
          a6 = 1;
          for ( j = (v37 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v37 - 1) & v41 )
          {
            v40 = v38 + 24LL * j;
            a5 = *(void **)v40;
            if ( *(_UNKNOWN **)v40 == &unk_4F81450 && a2 == *(_QWORD *)(v40 + 8) )
              break;
            if ( a5 == (void *)-4096LL && *(_QWORD *)(v40 + 8) == -4096 )
              goto LABEL_57;
            v41 = a6 + j;
            a6 = (unsigned int)(a6 + 1);
          }
        }
        else
        {
LABEL_57:
          v40 = v38 + 24 * v37;
        }
        v46 = *(_QWORD *)(*(_QWORD *)(v40 + 16) + 24LL);
        v47 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v46 + 16LL);
        if ( v47 == sub_D00100 )
          v48 = sub_B19B20(v46 + 8, a2, a3, (__int64)a4);
        else
          v48 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *, void *, __int64))v47)(
                  v46,
                  a2,
                  a3,
                  a4,
                  a5,
                  a6);
        v52[0] = v48;
        v51 = &unk_4F81450;
        sub_BBCF50((__int64)v53, v18, (__int64 *)&v51, v52);
        v24 = v54;
      }
      return *(unsigned __int8 *)(v24 + 8);
    }
    v49 = 1;
    while ( a5 != (void *)-4096LL )
    {
      a6 = (unsigned int)(v49 + 1);
      v23 = v21 & (v49 + v23);
      v24 = v20 + 16LL * v23;
      a5 = *(void **)v24;
      if ( *(_UNKNOWN **)v24 == &unk_4F81450 )
        goto LABEL_14;
      v49 = a6;
    }
    if ( v19 )
    {
      v45 = 128;
      goto LABEL_45;
    }
    v22 = *(unsigned int *)(v18 + 24);
LABEL_44:
    v45 = 16 * v22;
LABEL_45:
    v24 = v20 + v45;
    goto LABEL_14;
  }
  return result;
}
