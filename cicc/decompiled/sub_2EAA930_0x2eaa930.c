// Function: sub_2EAA930
// Address: 0x2eaa930
//
_QWORD *__fastcall sub_2EAA930(_QWORD *a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  __int64 v6; // r8
  int v7; // r11d
  __int64 v8; // rcx
  _QWORD *v9; // r14
  unsigned int v10; // edx
  _QWORD *v11; // rax
  __int64 v12; // r9
  _QWORD *v13; // r13
  int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // r10
  __int64 (*v22)(); // rax
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r10
  __int64 v26; // rsi
  void (*v27)(); // rdx
  __int64 v28; // rcx
  void (*v29)(); // rax
  unsigned __int64 v30; // r15
  int v31; // eax
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  _QWORD *v35; // rdi
  unsigned int v36; // r13d
  int v37; // r9d
  __int64 v38; // [rsp+0h] [rbp-40h]
  int v39; // [rsp+8h] [rbp-38h]

  if ( a1[318] == a2 )
    return (_QWORD *)a1[319];
  v4 = *((_DWORD *)a1 + 632);
  v5 = (__int64)(a1 + 313);
  if ( !v4 )
  {
    ++a1[313];
    goto LABEL_9;
  }
  v6 = v4 - 1;
  v7 = 1;
  v8 = a1[314];
  v9 = 0;
  v10 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (_QWORD *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( a2 != *v11 )
  {
    while ( v12 != -4096 )
    {
      if ( v12 == -8192 && !v9 )
        v9 = v11;
      v10 = v6 & (v7 + v10);
      v11 = (_QWORD *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
        goto LABEL_4;
      ++v7;
    }
    if ( !v9 )
      v9 = v11;
    v31 = *((_DWORD *)a1 + 630);
    ++a1[313];
    v19 = (unsigned int)(v31 + 1);
    if ( 4 * (int)v19 < 3 * v4 )
    {
      v16 = v4 >> 3;
      if ( v4 - *((_DWORD *)a1 + 631) - (unsigned int)v19 > (unsigned int)v16 )
      {
LABEL_11:
        *((_DWORD *)a1 + 630) = v19;
        if ( *v9 != -4096 )
          --*((_DWORD *)a1 + 631);
        *v9 = a2;
        v21 = 0;
        v9[1] = 0;
        v22 = *(__int64 (**)())(*(_QWORD *)*a1 + 16LL);
        if ( v22 != sub_23CE270 )
          v21 = ((__int64 (__fastcall *)(_QWORD, __int64, __int64, __int64, __int64))v22)(*a1, a2, v19, v16, v6);
        v23 = a1[310];
        v38 = v21;
        v39 = *((_DWORD *)a1 + 634);
        if ( !v23 )
          v23 = (__int64)(a1 + 1);
        ++*((_DWORD *)a1 + 634);
        v24 = sub_22077B0(0x460u);
        v25 = v38;
        v13 = (_QWORD *)v24;
        if ( v24 )
        {
          sub_2E81B70(v24, a2, *a1, v38, v23, v39);
          v25 = v38;
        }
        v26 = v25;
        sub_2E78D90(v13, v25);
        v29 = *(void (**)())(*(_QWORD *)*a1 + 248LL);
        if ( v29 != nullsub_1497 )
        {
          v26 = (__int64)v13;
          ((void (__fastcall *)(_QWORD, _QWORD *))v29)(*a1, v13);
        }
        v30 = v9[1];
        v9[1] = v13;
        if ( v30 )
        {
          sub_2E81F20(v30, v26, v27, v28);
          j_j___libc_free_0(v30);
        }
        goto LABEL_5;
      }
      sub_2EAA720(v5, v4);
      v32 = *((_DWORD *)a1 + 632);
      if ( v32 )
      {
        v33 = v32 - 1;
        v34 = a1[314];
        v35 = 0;
        v36 = v33 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v6 = 1;
        v19 = (unsigned int)(*((_DWORD *)a1 + 630) + 1);
        v9 = (_QWORD *)(v34 + 16LL * v36);
        v16 = *v9;
        if ( a2 != *v9 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v35 )
              v35 = v9;
            v36 = v33 & (v6 + v36);
            v9 = (_QWORD *)(v34 + 16LL * v36);
            v16 = *v9;
            if ( a2 == *v9 )
              goto LABEL_11;
            v6 = (unsigned int)(v6 + 1);
          }
          if ( v35 )
            v9 = v35;
        }
        goto LABEL_11;
      }
LABEL_54:
      ++*((_DWORD *)a1 + 630);
      BUG();
    }
LABEL_9:
    sub_2EAA720(v5, 2 * v4);
    v15 = *((_DWORD *)a1 + 632);
    if ( v15 )
    {
      v16 = (unsigned int)(v15 - 1);
      v17 = a1[314];
      v18 = v16 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = (unsigned int)(*((_DWORD *)a1 + 630) + 1);
      v9 = (_QWORD *)(v17 + 16LL * v18);
      v20 = *v9;
      if ( a2 != *v9 )
      {
        v37 = 1;
        v6 = 0;
        while ( v20 != -4096 )
        {
          if ( !v6 && v20 == -8192 )
            v6 = (__int64)v9;
          v18 = v16 & (v37 + v18);
          v9 = (_QWORD *)(v17 + 16LL * v18);
          v20 = *v9;
          if ( a2 == *v9 )
            goto LABEL_11;
          ++v37;
        }
        if ( v6 )
          v9 = (_QWORD *)v6;
      }
      goto LABEL_11;
    }
    goto LABEL_54;
  }
LABEL_4:
  v13 = (_QWORD *)v11[1];
LABEL_5:
  a1[318] = a2;
  a1[319] = v13;
  return v13;
}
