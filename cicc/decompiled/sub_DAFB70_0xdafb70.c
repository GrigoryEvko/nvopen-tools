// Function: sub_DAFB70
// Address: 0xdafb70
//
_QWORD *__fastcall sub_DAFB70(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int64 *v7; // r13
  _DWORD *v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // r12
  unsigned __int64 *v11; // rbx
  int v12; // r10d
  unsigned __int64 v13; // r12
  __int64 v14; // rdx
  unsigned __int64 v15; // r12
  int v16; // eax
  _DWORD *v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // r12
  __int64 v20; // rax
  _QWORD *v21; // r12
  __int64 v23; // rdx
  unsigned __int64 *v24; // r10
  unsigned __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // r13
  char *v28; // rbx
  __int64 v29; // rdx
  __int16 v30; // r15
  unsigned int v31; // esi
  __int64 v32; // rdx
  __int64 v33; // r9
  __int64 v34; // rdi
  int v35; // r11d
  unsigned int v36; // eax
  __int64 *v37; // r15
  __int64 v38; // rcx
  __int64 v39; // rax
  _QWORD *v40; // r12
  __int64 v41; // rax
  int v42; // eax
  int v43; // ecx
  _QWORD *v44; // rax
  __int64 v45; // [rsp+20h] [rbp-150h]
  void *v46; // [rsp+28h] [rbp-148h]
  unsigned __int64 *v47; // [rsp+30h] [rbp-140h]
  __int64 *v48; // [rsp+30h] [rbp-140h]
  __int64 v49; // [rsp+30h] [rbp-140h]
  __int64 n; // [rsp+40h] [rbp-130h]
  size_t na; // [rsp+40h] [rbp-130h]
  int desta; // [rsp+48h] [rbp-128h]
  char *dest; // [rsp+48h] [rbp-128h]
  int v55; // [rsp+5Ch] [rbp-114h]
  __int64 v57; // [rsp+68h] [rbp-108h] BYREF
  __int64 *v58; // [rsp+78h] [rbp-F8h] BYREF
  __int64 v59; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v60; // [rsp+88h] [rbp-E8h]
  __int64 v61; // [rsp+90h] [rbp-E0h] BYREF
  unsigned int v62; // [rsp+98h] [rbp-D8h]
  _QWORD *v63; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v64; // [rsp+A8h] [rbp-C8h]
  _DWORD *v65; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+B8h] [rbp-B8h]
  _DWORD v67[44]; // [rsp+C0h] [rbp-B0h] BYREF

  v6 = a1;
  v66 = 0x2000000001LL;
  v7 = &a2[a3];
  v57 = a4;
  v55 = a5;
  v65 = v67;
  v67[0] = 8;
  n = 8 * a3;
  if ( a2 == v7 )
  {
    v15 = v57;
    v18 = 1;
    v17 = v67;
    v16 = v57;
  }
  else
  {
    v8 = v67;
    v9 = 1;
    v10 = *a2;
    v11 = a2 + 1;
    v12 = *a2;
    while ( 1 )
    {
      v8[v9] = v12;
      v13 = HIDWORD(v10);
      v14 = (unsigned int)(v66 + 1);
      a5 = v14 + 1;
      LODWORD(v66) = v66 + 1;
      if ( v14 + 1 > (unsigned __int64)HIDWORD(v66) )
      {
        sub_C8D5F0((__int64)&v65, v67, v14 + 1, 4u, a5, a6);
        v14 = (unsigned int)v66;
      }
      v65[v14] = v13;
      v9 = (unsigned int)(v66 + 1);
      LODWORD(v66) = v66 + 1;
      if ( v7 == v11 )
        break;
      v10 = *v11;
      v12 = *v11;
      if ( v9 + 1 > (unsigned __int64)HIDWORD(v66) )
      {
        desta = *v11;
        sub_C8D5F0((__int64)&v65, v67, v9 + 1, 4u, v9 + 1, a6);
        v9 = (unsigned int)v66;
        v12 = desta;
      }
      v8 = v65;
      ++v11;
    }
    v15 = v57;
    v6 = a1;
    v16 = v57;
    if ( v9 + 1 > (unsigned __int64)HIDWORD(v66) )
    {
      sub_C8D5F0((__int64)&v65, v67, v9 + 1, 4u, a5, a6);
      v17 = v65;
      v16 = v57;
      v18 = (unsigned int)v66;
    }
    else
    {
      v17 = v65;
      v18 = v9;
    }
  }
  v17[v18] = v16;
  v19 = HIDWORD(v15);
  LODWORD(v66) = v66 + 1;
  v20 = (unsigned int)v66;
  if ( (unsigned __int64)(unsigned int)v66 + 1 > HIDWORD(v66) )
  {
    sub_C8D5F0((__int64)&v65, v67, (unsigned int)v66 + 1LL, 4u, a5, a6);
    v20 = (unsigned int)v66;
  }
  v65[v20] = v19;
  LODWORD(v66) = v66 + 1;
  v58 = 0;
  v21 = sub_C65B40(v6 + 1032, (__int64)&v65, (__int64 *)&v58, (__int64)off_49DEA80);
  if ( !v21 )
  {
    v23 = *(_QWORD *)(v6 + 1064);
    v24 = (unsigned __int64 *)(v6 + 1064);
    *(_QWORD *)(v6 + 1144) += n;
    v25 = n + ((v23 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( *(_QWORD *)(v6 + 1072) >= v25 && v23 )
    {
      *(_QWORD *)(v6 + 1064) = v25;
      dest = (char *)((v23 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v41 = sub_9D1E70(v6 + 1064, n, n, 3);
      v24 = (unsigned __int64 *)(v6 + 1064);
      dest = (char *)v41;
    }
    if ( a2 != v7 )
    {
      v47 = v24;
      memmove(dest, a2, n);
      v24 = v47;
    }
    v48 = (__int64 *)v24;
    na = (size_t)&dest[n];
    v46 = sub_C65D30((__int64)&v65, v24);
    v45 = v26;
    v27 = sub_A777F0(0x38u, v48);
    if ( v27 )
    {
      v60 = 16;
      v59 = 1;
      v49 = v57;
      if ( (char *)na == dest )
        goto LABEL_42;
      v28 = dest;
      do
      {
        v29 = *(unsigned __int16 *)(*(_QWORD *)v28 + 26LL);
        v64 = 16;
        v63 = (_QWORD *)v29;
        sub_C49B30((__int64)&v61, (__int64)&v59, (__int64 *)&v63);
        if ( v60 > 0x40 && v59 )
          j_j___libc_free_0_0(v59);
        v59 = v61;
        v60 = v62;
        if ( v64 > 0x40 && v63 )
          j_j___libc_free_0_0(v63);
        v28 += 8;
      }
      while ( (char *)na != v28 );
      v21 = 0;
      if ( v60 <= 0x40 )
      {
LABEL_42:
        v30 = v59;
      }
      else
      {
        v30 = *(_WORD *)v59;
        j_j___libc_free_0_0(v59);
      }
      *(_WORD *)(v27 + 26) = v30;
      *(_QWORD *)v27 = 0;
      *(_QWORD *)(v27 + 8) = v46;
      *(_WORD *)(v27 + 28) = 0;
      *(_QWORD *)(v27 + 16) = v45;
      *(_WORD *)(v27 + 24) = 8;
      *(_QWORD *)(v27 + 32) = dest;
      *(_QWORD *)(v27 + 40) = a3;
      *(_QWORD *)(v27 + 48) = v49;
    }
    sub_C657C0((__int64 *)(v6 + 1032), (__int64 *)v27, v58, (__int64)off_49DEA80);
    v31 = *(_DWORD *)(v6 + 1184);
    if ( v31 )
    {
      v32 = v57;
      v33 = v31 - 1;
      v34 = *(_QWORD *)(v6 + 1168);
      v35 = 1;
      v36 = v33 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v37 = (__int64 *)(v34 + 56LL * v36);
      v38 = *v37;
      if ( *v37 == v57 )
      {
LABEL_37:
        v39 = *((unsigned int *)v37 + 4);
        v40 = v37 + 1;
        if ( v39 + 1 > (unsigned __int64)*((unsigned int *)v37 + 5) )
        {
          sub_C8D5F0((__int64)(v37 + 1), v37 + 3, v39 + 1, 8u, v6 + 1160, v33);
          v39 = *((unsigned int *)v37 + 4);
        }
LABEL_39:
        *(_QWORD *)(*v40 + 8 * v39) = v27;
        ++*((_DWORD *)v40 + 2);
        v21 = (_QWORD *)v27;
        sub_DAEE00(v6, v27, (__int64 *)a2, a3);
        goto LABEL_14;
      }
      while ( v38 != -4096 )
      {
        if ( !v21 && v38 == -8192 )
          v21 = v37;
        v36 = v33 & (v36 + v35);
        v37 = (__int64 *)(v34 + 56LL * v36);
        v38 = *v37;
        if ( v57 == *v37 )
          goto LABEL_37;
        ++v35;
      }
      v42 = *(_DWORD *)(v6 + 1176);
      if ( !v21 )
        v21 = v37;
      ++*(_QWORD *)(v6 + 1160);
      v43 = v42 + 1;
      v63 = v21;
      if ( 4 * (v42 + 1) < 3 * v31 )
      {
        if ( v31 - *(_DWORD *)(v6 + 1180) - v43 > v31 >> 3 )
        {
LABEL_54:
          *(_DWORD *)(v6 + 1176) = v43;
          if ( *v21 != -4096 )
            --*(_DWORD *)(v6 + 1180);
          v44 = v21 + 3;
          *v21 = v32;
          v40 = v21 + 1;
          *v40 = v44;
          v40[1] = 0x400000000LL;
          v39 = 0;
          goto LABEL_39;
        }
LABEL_59:
        sub_DA5EE0(v6 + 1160, v31);
        sub_D9E4A0(v6 + 1160, &v57, &v63);
        v32 = v57;
        v21 = v63;
        v43 = *(_DWORD *)(v6 + 1176) + 1;
        goto LABEL_54;
      }
    }
    else
    {
      ++*(_QWORD *)(v6 + 1160);
      v63 = 0;
    }
    v31 *= 2;
    goto LABEL_59;
  }
LABEL_14:
  sub_D97270(v6, (__int64)v21, v55);
  if ( v65 != v67 )
    _libc_free(v65, v21);
  return v21;
}
