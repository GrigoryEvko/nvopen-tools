// Function: sub_39C5FF0
// Address: 0x39c5ff0
//
__int64 __fastcall sub_39C5FF0(__int64 a1, unsigned __int16 a2, int a3, __int64 a4)
{
  unsigned __int64 v4; // rdx
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *v13; // rbx
  __int64 v14; // rax
  unsigned int v15; // edx
  int v17; // r10d
  __int64 *v18; // r9
  int v19; // eax
  int v20; // edx
  int *v21; // rax
  size_t v22; // rdx
  int v23; // eax
  int v24; // ecx
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rsi
  int v28; // r9d
  __int64 *v29; // r8
  int v30; // eax
  int v31; // eax
  __int64 v32; // rsi
  int v33; // r8d
  unsigned int v34; // r15d
  __int64 *v35; // rdi
  __int64 v36; // rcx

  v4 = (unsigned int)(a3 - 15);
  if ( (unsigned __int16)v4 > 0x33u
    || (v8 = 0x8000000010003LL, !_bittest64(&v8, v4))
    || a2 != 73
    || (v21 = (int *)sub_39C2E60(a4), !v22) )
  {
    v9 = *(_DWORD *)(a1 + 184);
    v10 = a1 + 160;
    if ( v9 )
    {
      v11 = *(_QWORD *)(a1 + 168);
      v12 = (v9 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( a4 == *v13 )
      {
LABEL_6:
        v15 = *((_DWORD *)v13 + 2);
        if ( v15 )
          return sub_39C5AC0((int *)a1, a2, v15);
        goto LABEL_17;
      }
      v17 = 1;
      v18 = 0;
      while ( v14 != -8 )
      {
        if ( v14 == -16 && !v18 )
          v18 = v13;
        v12 = (v9 - 1) & (v17 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( a4 == *v13 )
          goto LABEL_6;
        ++v17;
      }
      v19 = *(_DWORD *)(a1 + 176);
      if ( v18 )
        v13 = v18;
      ++*(_QWORD *)(a1 + 160);
      v20 = v19 + 1;
      if ( 4 * (v19 + 1) < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(a1 + 180) - v20 > v9 >> 3 )
        {
LABEL_14:
          *(_DWORD *)(a1 + 176) = v20;
          if ( *v13 != -8 )
            --*(_DWORD *)(a1 + 180);
          *v13 = a4;
          *((_DWORD *)v13 + 2) = 0;
LABEL_17:
          sub_39C2ED0((int *)a1, 0x54u);
          sub_39C2ED0((int *)a1, a2);
          *((_DWORD *)v13 + 2) = *(_DWORD *)(a1 + 176);
          return sub_39C6FD0(a1, a4);
        }
        sub_39C5E30(v10, v9);
        v30 = *(_DWORD *)(a1 + 184);
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *(_QWORD *)(a1 + 168);
          v33 = 1;
          v34 = v31 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
          v20 = *(_DWORD *)(a1 + 176) + 1;
          v35 = 0;
          v13 = (__int64 *)(v32 + 16LL * v34);
          v36 = *v13;
          if ( a4 != *v13 )
          {
            while ( v36 != -8 )
            {
              if ( !v35 && v36 == -16 )
                v35 = v13;
              v34 = v31 & (v33 + v34);
              v13 = (__int64 *)(v32 + 16LL * v34);
              v36 = *v13;
              if ( a4 == *v13 )
                goto LABEL_14;
              ++v33;
            }
            if ( v35 )
              v13 = v35;
          }
          goto LABEL_14;
        }
LABEL_49:
        ++*(_DWORD *)(a1 + 176);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 160);
    }
    sub_39C5E30(v10, 2 * v9);
    v23 = *(_DWORD *)(a1 + 184);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 168);
      v26 = (v23 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v20 = *(_DWORD *)(a1 + 176) + 1;
      v13 = (__int64 *)(v25 + 16LL * v26);
      v27 = *v13;
      if ( a4 != *v13 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -8 )
        {
          if ( !v29 && v27 == -16 )
            v29 = v13;
          v26 = v24 & (v28 + v26);
          v13 = (__int64 *)(v25 + 16LL * v26);
          v27 = *v13;
          if ( a4 == *v13 )
            goto LABEL_14;
          ++v28;
        }
        if ( v29 )
          v13 = v29;
      }
      goto LABEL_14;
    }
    goto LABEL_49;
  }
  return sub_39C5A40((int *)a1, 0x49u, a4, v21, v22);
}
