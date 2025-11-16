// Function: sub_34C3D00
// Address: 0x34c3d00
//
__int64 __fastcall sub_34C3D00(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4)
{
  __int64 (*v8)(); // rax
  __int64 v9; // rax
  __int64 *v10; // r15
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // r8
  int v24; // eax
  __int64 v25; // rcx
  int v26; // esi
  unsigned int v27; // edx
  _QWORD *v28; // rax
  _QWORD *v29; // rdi
  __int64 *v30; // rdi
  __int64 v31; // r14
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 v34; // rcx
  unsigned int v35; // edi
  unsigned int v36; // edx
  __int64 v37; // rax
  _QWORD *v38; // r8
  int v39; // ebx
  int v40; // r11d
  __int64 *v41; // r8
  unsigned int v42; // r14d
  unsigned int v43; // r9d
  __int64 *v44; // rax
  __int64 v45; // rdx
  int v47; // eax
  int v48; // r9d
  __int64 v49; // rdi
  int v50; // eax
  int v51; // edx
  int v52; // eax
  int v53; // r9d
  int v54; // eax
  int v55; // ecx
  __int64 v56; // rsi
  unsigned int v57; // r14d
  __int64 v58; // rax
  int v59; // r9d
  __int64 *v60; // rdi
  int v61; // eax
  int v62; // ecx
  __int64 v63; // rsi
  int v64; // r9d
  unsigned int v65; // r14d
  __int64 v66; // rax
  __int64 v67; // [rsp+14h] [rbp-3Ch]

  v8 = *(__int64 (**)())(**(_QWORD **)(a1 + 136) + 408LL);
  if ( v8 != sub_2FDC540 && !(unsigned __int8)v8() )
    return 0;
  v9 = sub_2E7AAE0(a2[4], a4, v67, 0);
  v10 = (__int64 *)a2[1];
  v11 = v9;
  sub_2E33BD0(a2[4] + 320LL, v9);
  v12 = *v10;
  v13 = *(_QWORD *)v11;
  *(_QWORD *)(v11 + 8) = v10;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v11 = v12 | v13 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  *v10 = v11 | *v10 & 7;
  v14 = a2 + 6;
  sub_2E340B0(v11, (__int64)a2, v12, v15, v16, v17);
  sub_2E33F80((__int64)a2, v11, -1, v18, v19, v20);
  if ( a2 + 6 != a3 && v14 != (__int64 *)(v11 + 48) )
  {
    sub_2E310C0((__int64 *)(v11 + 40), a2 + 5, (__int64)a3, (__int64)(a2 + 6));
    if ( v14 != a3 )
    {
      v21 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v14;
      a2[6] = a2[6] & 7LL | *a3 & 0xFFFFFFFFFFFFFFF8LL;
      v22 = *(_QWORD *)(v11 + 48);
      *(_QWORD *)(v21 + 8) = v11 + 48;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      *a3 = v22 | *a3 & 7;
      *(_QWORD *)(v22 + 8) = a3;
      *(_QWORD *)(v11 + 48) = v21 | *(_QWORD *)(v11 + 48) & 7LL;
    }
  }
  v23 = *(_QWORD *)(a1 + 160);
  if ( v23 )
  {
    v24 = *(_DWORD *)(v23 + 24);
    v25 = *(_QWORD *)(v23 + 8);
    if ( v24 )
    {
      v26 = v24 - 1;
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = (_QWORD *)(v25 + 16LL * v27);
      v29 = (_QWORD *)*v28;
      if ( a2 == (_QWORD *)*v28 )
      {
LABEL_9:
        v30 = (__int64 *)v28[1];
        if ( v30 )
          sub_2EA77F0(v30, v11, *(_QWORD *)(a1 + 160));
      }
      else
      {
        v52 = 1;
        while ( v29 != (_QWORD *)-4096LL )
        {
          v53 = v52 + 1;
          v27 = v26 & (v52 + v27);
          v28 = (_QWORD *)(v25 + 16LL * v27);
          v29 = (_QWORD *)*v28;
          if ( a2 == (_QWORD *)*v28 )
            goto LABEL_9;
          v52 = v53;
        }
      }
    }
  }
  v31 = *(_QWORD *)(a1 + 232);
  v32 = sub_2F06CB0(v31, (__int64)a2);
  sub_2F06FC0(v31, v11, v32);
  if ( *(_BYTE *)(a1 + 131) )
    sub_3509790(a1 + 168, v11);
  v33 = *(_DWORD *)(a1 + 96);
  v34 = *(_QWORD *)(a1 + 80);
  if ( v33 )
  {
    v35 = v33 - 1;
    v36 = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v37 = v34 + 16LL * v36;
    v38 = *(_QWORD **)v37;
    if ( a2 == *(_QWORD **)v37 )
    {
LABEL_15:
      if ( v37 == v34 + 16LL * v33 )
        return v11;
      v39 = *(_DWORD *)(v37 + 8);
      v40 = 1;
      v41 = 0;
      v42 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
      v43 = v42 & v35;
      v44 = (__int64 *)(v34 + 16LL * (v42 & v35));
      v45 = *v44;
      if ( v11 == *v44 )
      {
LABEL_17:
        *((_DWORD *)v44 + 2) = v39;
        return v11;
      }
      while ( v45 != -4096 )
      {
        if ( v45 == -8192 && !v41 )
          v41 = v44;
        v43 = v35 & (v40 + v43);
        v44 = (__int64 *)(v34 + 16LL * v43);
        v45 = *v44;
        if ( v11 == *v44 )
          goto LABEL_17;
        ++v40;
      }
      v49 = a1 + 72;
      if ( !v41 )
        v41 = v44;
      v50 = *(_DWORD *)(a1 + 88);
      ++*(_QWORD *)(a1 + 72);
      v51 = v50 + 1;
      if ( 4 * (v50 + 1) >= 3 * v33 )
      {
        sub_2E3ADF0(v49, 2 * v33);
        v54 = *(_DWORD *)(a1 + 96);
        if ( v54 )
        {
          v55 = v54 - 1;
          v56 = *(_QWORD *)(a1 + 80);
          v57 = (v54 - 1) & v42;
          v51 = *(_DWORD *)(a1 + 88) + 1;
          v41 = (__int64 *)(v56 + 16LL * v57);
          v58 = *v41;
          if ( v11 == *v41 )
            goto LABEL_35;
          v59 = 1;
          v60 = 0;
          while ( v58 != -4096 )
          {
            if ( !v60 && v58 == -8192 )
              v60 = v41;
            v57 = v55 & (v59 + v57);
            v41 = (__int64 *)(v56 + 16LL * v57);
            v58 = *v41;
            if ( v11 == *v41 )
              goto LABEL_35;
            ++v59;
          }
LABEL_46:
          if ( v60 )
            v41 = v60;
          goto LABEL_35;
        }
      }
      else
      {
        if ( v33 - *(_DWORD *)(a1 + 92) - v51 > v33 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 88) = v51;
          if ( *v41 != -4096 )
            --*(_DWORD *)(a1 + 92);
          *v41 = v11;
          *((_DWORD *)v41 + 2) = 0;
          *((_DWORD *)v41 + 2) = v39;
          return v11;
        }
        sub_2E3ADF0(v49, v33);
        v61 = *(_DWORD *)(a1 + 96);
        if ( v61 )
        {
          v62 = v61 - 1;
          v63 = *(_QWORD *)(a1 + 80);
          v64 = 1;
          v65 = (v61 - 1) & v42;
          v51 = *(_DWORD *)(a1 + 88) + 1;
          v60 = 0;
          v41 = (__int64 *)(v63 + 16LL * v65);
          v66 = *v41;
          if ( v11 == *v41 )
            goto LABEL_35;
          while ( v66 != -4096 )
          {
            if ( v66 == -8192 && !v60 )
              v60 = v41;
            v65 = v62 & (v64 + v65);
            v41 = (__int64 *)(v63 + 16LL * v65);
            v66 = *v41;
            if ( v11 == *v41 )
              goto LABEL_35;
            ++v64;
          }
          goto LABEL_46;
        }
      }
      ++*(_DWORD *)(a1 + 88);
      BUG();
    }
    v47 = 1;
    while ( v38 != (_QWORD *)-4096LL )
    {
      v48 = v47 + 1;
      v36 = v35 & (v47 + v36);
      v37 = v34 + 16LL * v36;
      v38 = *(_QWORD **)v37;
      if ( a2 == *(_QWORD **)v37 )
        goto LABEL_15;
      v47 = v48;
    }
  }
  return v11;
}
