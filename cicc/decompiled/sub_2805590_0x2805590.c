// Function: sub_2805590
// Address: 0x2805590
//
unsigned __int8 *__fastcall sub_2805590(unsigned __int8 *a1, __int64 a2, __m128i *a3)
{
  unsigned __int8 v4; // cl
  __int64 v6; // rdi
  unsigned int v8; // esi
  unsigned int v10; // edx
  unsigned __int8 **v11; // rax
  unsigned __int8 *v12; // r9
  __int64 *v13; // r14
  __int64 *v14; // rax
  __int64 v15; // r14
  int v16; // r11d
  unsigned __int8 **v17; // rdx
  unsigned int v18; // ecx
  _QWORD *v19; // rax
  unsigned __int8 *v20; // r9
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  _BYTE *v24; // rdi
  __int64 v25; // r8
  __int64 v26; // rdi
  _BYTE *v27; // r14
  _BYTE *v28; // rax
  int v29; // eax
  int v30; // ecx
  int v31; // eax
  int v32; // r10d
  int v33; // eax
  int v34; // edi
  __int64 v35; // r8
  unsigned int v36; // esi
  unsigned __int8 *v37; // rax
  int v38; // r10d
  unsigned __int8 **v39; // r9
  int v40; // eax
  int v41; // esi
  __int64 v42; // rdi
  int v43; // r9d
  unsigned int v44; // r13d
  unsigned __int8 **v45; // r8
  unsigned __int8 *v46; // rax

  v4 = *a1;
  if ( *a1 <= 0x1Cu )
    return a1;
  v6 = *(_QWORD *)(a2 + 8);
  v8 = *(_DWORD *)(a2 + 24);
  if ( v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v11 = (unsigned __int8 **)(v6 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a1 )
    {
LABEL_5:
      if ( v11 != (unsigned __int8 **)(v6 + 16LL * v8) )
        return v11[1];
    }
    else
    {
      v31 = 1;
      while ( v12 != (unsigned __int8 *)-4096LL )
      {
        v32 = v31 + 1;
        v10 = (v8 - 1) & (v31 + v10);
        v11 = (unsigned __int8 **)(v6 + 16LL * v10);
        v12 = *v11;
        if ( *v11 == a1 )
          goto LABEL_5;
        v31 = v32;
      }
    }
  }
  if ( (unsigned int)v4 - 42 > 0x11 )
  {
    if ( v4 == 82 )
    {
      v27 = (_BYTE *)sub_2805590(*((_QWORD *)a1 - 8), a2, a3);
      v28 = (_BYTE *)sub_2805590(*((_QWORD *)a1 - 4), a2, a3);
      v15 = sub_1016CC0(*((_WORD *)a1 + 1) & 0x3F, v27, v28, a3->m128i_i64);
    }
    else
    {
      v15 = (__int64)a1;
      if ( v4 != 86 )
        goto LABEL_11;
      v24 = (_BYTE *)sub_2805590(*((_QWORD *)a1 - 12), a2, a3);
      if ( *v24 != 17 )
      {
        v6 = *(_QWORD *)(a2 + 8);
        v8 = *(_DWORD *)(a2 + 24);
        goto LABEL_11;
      }
      if ( sub_AD7930(v24, a2, v22, v23, v25) )
        v26 = *((_QWORD *)a1 - 8);
      else
        v26 = *((_QWORD *)a1 - 4);
      v15 = sub_2805590(v26, a2, a3);
    }
  }
  else
  {
    v13 = (__int64 *)sub_2805590(*((_QWORD *)a1 - 8), a2, a3);
    v14 = (__int64 *)sub_2805590(*((_QWORD *)a1 - 4), a2, a3);
    v15 = (__int64)sub_101E7C0((unsigned int)*a1 - 29, v13, v14, a3);
  }
  v6 = *(_QWORD *)(a2 + 8);
  v8 = *(_DWORD *)(a2 + 24);
  if ( !v15 )
    v15 = (__int64)a1;
LABEL_11:
  if ( !v8 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_40;
  }
  v16 = 1;
  v17 = 0;
  v18 = (v8 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v19 = (_QWORD *)(v6 + 16LL * v18);
  v20 = (unsigned __int8 *)*v19;
  if ( (unsigned __int8 *)*v19 != a1 )
  {
    while ( v20 != (unsigned __int8 *)-4096LL )
    {
      if ( v20 == (unsigned __int8 *)-8192LL && !v17 )
        v17 = (unsigned __int8 **)v19;
      v18 = (v8 - 1) & (v16 + v18);
      v19 = (_QWORD *)(v6 + 16LL * v18);
      v20 = (unsigned __int8 *)*v19;
      if ( (unsigned __int8 *)*v19 == a1 )
        goto LABEL_13;
      ++v16;
    }
    if ( !v17 )
      v17 = (unsigned __int8 **)v19;
    v29 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v30 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v8 )
    {
      if ( v8 - (v30 + *(_DWORD *)(a2 + 20)) > v8 >> 3 )
      {
LABEL_32:
        *(_DWORD *)(a2 + 16) = v30;
        if ( *v17 != (unsigned __int8 *)-4096LL )
          --*(_DWORD *)(a2 + 20);
        *v17 = a1;
        v21 = (__int64 *)(v17 + 1);
        v17[1] = 0;
        goto LABEL_14;
      }
      sub_FAA400(a2, v8);
      v40 = *(_DWORD *)(a2 + 24);
      if ( v40 )
      {
        v41 = v40 - 1;
        v42 = *(_QWORD *)(a2 + 8);
        v43 = 1;
        v44 = (v40 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v45 = 0;
        v30 = *(_DWORD *)(a2 + 16) + 1;
        v17 = (unsigned __int8 **)(v42 + 16LL * v44);
        v46 = *v17;
        if ( *v17 != a1 )
        {
          while ( v46 != (unsigned __int8 *)-4096LL )
          {
            if ( v46 == (unsigned __int8 *)-8192LL && !v45 )
              v45 = v17;
            v44 = v41 & (v43 + v44);
            v17 = (unsigned __int8 **)(v42 + 16LL * v44);
            v46 = *v17;
            if ( *v17 == a1 )
              goto LABEL_32;
            ++v43;
          }
          if ( v45 )
            v17 = v45;
        }
        goto LABEL_32;
      }
LABEL_65:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
LABEL_40:
    sub_FAA400(a2, 2 * v8);
    v33 = *(_DWORD *)(a2 + 24);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a2 + 8);
      v36 = (v33 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v30 = *(_DWORD *)(a2 + 16) + 1;
      v17 = (unsigned __int8 **)(v35 + 16LL * v36);
      v37 = *v17;
      if ( *v17 != a1 )
      {
        v38 = 1;
        v39 = 0;
        while ( v37 != (unsigned __int8 *)-4096LL )
        {
          if ( !v39 && v37 == (unsigned __int8 *)-8192LL )
            v39 = v17;
          v36 = v34 & (v38 + v36);
          v17 = (unsigned __int8 **)(v35 + 16LL * v36);
          v37 = *v17;
          if ( *v17 == a1 )
            goto LABEL_32;
          ++v38;
        }
        if ( v39 )
          v17 = v39;
      }
      goto LABEL_32;
    }
    goto LABEL_65;
  }
LABEL_13:
  v21 = v19 + 1;
LABEL_14:
  *v21 = v15;
  return (unsigned __int8 *)v15;
}
