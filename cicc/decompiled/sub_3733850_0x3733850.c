// Function: sub_3733850
// Address: 0x3733850
//
__int64 __fastcall sub_3733850(__int64 a1, unsigned __int16 a2, int a3, __int64 a4)
{
  unsigned __int64 v4; // rdx
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // rdi
  int v11; // r10d
  __int64 v12; // r8
  __int64 *v13; // r14
  unsigned int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // edx
  _DWORD *v18; // r14
  int v20; // eax
  int v21; // edx
  int *v22; // rax
  size_t v23; // rdx
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rsi
  int v29; // r9d
  __int64 *v30; // r8
  int v31; // eax
  int v32; // eax
  __int64 v33; // rsi
  int v34; // r8d
  unsigned int v35; // r15d
  __int64 *v36; // rdi
  __int64 v37; // rcx

  v4 = (unsigned int)(a3 - 15);
  if ( (unsigned __int16)v4 > 0x33u
    || (v8 = 0x8000000010003LL, !_bittest64(&v8, v4))
    || a2 != 73
    || (v22 = (int *)sub_372FC20(a4), !v23) )
  {
    v9 = *(_DWORD *)(a1 + 192);
    v10 = a1 + 168;
    if ( v9 )
    {
      v11 = 1;
      v12 = *(_QWORD *)(a1 + 176);
      v13 = 0;
      v14 = (v9 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v15 = v12 + 16LL * v14;
      v16 = *(_QWORD *)v15;
      if ( a4 == *(_QWORD *)v15 )
      {
LABEL_6:
        v17 = *(_DWORD *)(v15 + 8);
        v18 = (_DWORD *)(v15 + 8);
        if ( v17 )
          return sub_3733290((int *)a1, a2, v17);
        goto LABEL_21;
      }
      while ( v16 != -4096 )
      {
        if ( v16 == -8192 && !v13 )
          v13 = (__int64 *)v15;
        v14 = (v9 - 1) & (v11 + v14);
        v15 = v12 + 16LL * v14;
        v16 = *(_QWORD *)v15;
        if ( a4 == *(_QWORD *)v15 )
          goto LABEL_6;
        ++v11;
      }
      if ( !v13 )
        v13 = (__int64 *)v15;
      v20 = *(_DWORD *)(a1 + 184);
      ++*(_QWORD *)(a1 + 168);
      v21 = v20 + 1;
      if ( 4 * (v20 + 1) < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(a1 + 188) - v21 > v9 >> 3 )
        {
LABEL_18:
          *(_DWORD *)(a1 + 184) = v21;
          if ( *v13 != -4096 )
            --*(_DWORD *)(a1 + 188);
          *v13 = a4;
          v18 = v13 + 1;
          *v18 = 0;
LABEL_21:
          sub_372FCB0((int *)a1, 0x54u);
          sub_372FCB0((int *)a1, a2);
          *v18 = *(_DWORD *)(a1 + 184);
          return sub_3734910(a1, a4);
        }
        sub_3733670(v10, v9);
        v31 = *(_DWORD *)(a1 + 192);
        if ( v31 )
        {
          v32 = v31 - 1;
          v33 = *(_QWORD *)(a1 + 176);
          v34 = 1;
          v35 = v32 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
          v21 = *(_DWORD *)(a1 + 184) + 1;
          v36 = 0;
          v13 = (__int64 *)(v33 + 16LL * v35);
          v37 = *v13;
          if ( a4 != *v13 )
          {
            while ( v37 != -4096 )
            {
              if ( !v36 && v37 == -8192 )
                v36 = v13;
              v35 = v32 & (v34 + v35);
              v13 = (__int64 *)(v33 + 16LL * v35);
              v37 = *v13;
              if ( a4 == *v13 )
                goto LABEL_18;
              ++v34;
            }
            if ( v36 )
              v13 = v36;
          }
          goto LABEL_18;
        }
LABEL_48:
        JUMPOUT(0x4379C0);
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 168);
    }
    sub_3733670(v10, 2 * v9);
    v24 = *(_DWORD *)(a1 + 192);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 176);
      v27 = (v24 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v21 = *(_DWORD *)(a1 + 184) + 1;
      v13 = (__int64 *)(v26 + 16LL * v27);
      v28 = *v13;
      if ( a4 != *v13 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -4096 )
        {
          if ( !v30 && v28 == -8192 )
            v30 = v13;
          v27 = v25 & (v29 + v27);
          v13 = (__int64 *)(v26 + 16LL * v27);
          v28 = *v13;
          if ( a4 == *v13 )
            goto LABEL_18;
          ++v29;
        }
        if ( v30 )
          v13 = v30;
      }
      goto LABEL_18;
    }
    goto LABEL_48;
  }
  return sub_3733210((int *)a1, 0x49u, a4, v22, v23);
}
