// Function: sub_3746590
// Address: 0x3746590
//
__int64 __fastcall sub_3746590(__int64 *a1, unsigned __int8 *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 (*v9)(); // rax
  unsigned int v10; // r12d
  unsigned int v12; // esi
  __int64 v13; // rdi
  int v14; // r15d
  unsigned __int8 **v15; // r10
  unsigned int v16; // ecx
  _QWORD *v17; // rax
  unsigned __int8 *v18; // rdx
  unsigned int *v19; // rax
  int v20; // eax
  int v21; // edx
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  unsigned int v25; // ecx
  unsigned __int8 *v26; // rsi
  int v27; // r9d
  unsigned __int8 **v28; // r8
  int v29; // eax
  int v30; // ecx
  __int64 v31; // rsi
  int v32; // r8d
  unsigned int v33; // r14d
  unsigned __int8 **v34; // rdi
  unsigned __int8 *v35; // rax

  if ( *a2 <= 0x15u && (v9 = *(__int64 (**)())(*a1 + 104), v9 != sub_3740E80) && (v10 = v9()) != 0
    || (v10 = sub_3746040(a1, a2, a3, a4, a5, a6)) != 0 )
  {
    v12 = *((_DWORD *)a1 + 8);
    if ( v12 )
    {
      v13 = a1[2];
      v14 = 1;
      v15 = 0;
      v16 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = (_QWORD *)(v13 + 16LL * v16);
      v18 = (unsigned __int8 *)*v17;
      if ( (unsigned __int8 *)*v17 == a2 )
      {
LABEL_8:
        v19 = (unsigned int *)(v17 + 1);
LABEL_9:
        *v19 = v10;
        a1[20] = sub_2EBEE10(a1[7], v10);
        return v10;
      }
      while ( v18 != (unsigned __int8 *)-4096LL )
      {
        if ( v18 == (unsigned __int8 *)-8192LL && !v15 )
          v15 = (unsigned __int8 **)v17;
        v16 = (v12 - 1) & (v14 + v16);
        v17 = (_QWORD *)(v13 + 16LL * v16);
        v18 = (unsigned __int8 *)*v17;
        if ( (unsigned __int8 *)*v17 == a2 )
          goto LABEL_8;
        ++v14;
      }
      if ( !v15 )
        v15 = (unsigned __int8 **)v17;
      v20 = *((_DWORD *)a1 + 6);
      ++a1[1];
      v21 = v20 + 1;
      if ( 4 * (v20 + 1) < 3 * v12 )
      {
        if ( v12 - *((_DWORD *)a1 + 7) - v21 > v12 >> 3 )
        {
LABEL_20:
          *((_DWORD *)a1 + 6) = v21;
          if ( *v15 != (unsigned __int8 *)-4096LL )
            --*((_DWORD *)a1 + 7);
          *v15 = a2;
          v19 = (unsigned int *)(v15 + 1);
          *((_DWORD *)v15 + 2) = 0;
          goto LABEL_9;
        }
        sub_3384500((__int64)(a1 + 1), v12);
        v29 = *((_DWORD *)a1 + 8);
        if ( v29 )
        {
          v30 = v29 - 1;
          v31 = a1[2];
          v32 = 1;
          v33 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v34 = 0;
          v21 = *((_DWORD *)a1 + 6) + 1;
          v15 = (unsigned __int8 **)(v31 + 16LL * v33);
          v35 = *v15;
          if ( *v15 != a2 )
          {
            while ( v35 != (unsigned __int8 *)-4096LL )
            {
              if ( !v34 && v35 == (unsigned __int8 *)-8192LL )
                v34 = v15;
              v33 = v30 & (v32 + v33);
              v15 = (unsigned __int8 **)(v31 + 16LL * v33);
              v35 = *v15;
              if ( *v15 == a2 )
                goto LABEL_20;
              ++v32;
            }
            if ( v34 )
              v15 = v34;
          }
          goto LABEL_20;
        }
LABEL_47:
        ++*((_DWORD *)a1 + 6);
        BUG();
      }
    }
    else
    {
      ++a1[1];
    }
    sub_3384500((__int64)(a1 + 1), 2 * v12);
    v22 = *((_DWORD *)a1 + 8);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = a1[2];
      v25 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = *((_DWORD *)a1 + 6) + 1;
      v15 = (unsigned __int8 **)(v24 + 16LL * v25);
      v26 = *v15;
      if ( *v15 != a2 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != (unsigned __int8 *)-4096LL )
        {
          if ( v26 == (unsigned __int8 *)-8192LL && !v28 )
            v28 = v15;
          v25 = v23 & (v27 + v25);
          v15 = (unsigned __int8 **)(v24 + 16LL * v25);
          v26 = *v15;
          if ( *v15 == a2 )
            goto LABEL_20;
          ++v27;
        }
        if ( v28 )
          v15 = v28;
      }
      goto LABEL_20;
    }
    goto LABEL_47;
  }
  return v10;
}
