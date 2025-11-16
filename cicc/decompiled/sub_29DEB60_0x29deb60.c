// Function: sub_29DEB60
// Address: 0x29deb60
//
void __fastcall sub_29DEB60(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  int v7; // r15d
  void *v8; // r14
  int v9; // r15d
  int v10; // eax
  size_t v11; // rdx
  int v12; // r8d
  unsigned int i; // ecx
  __int64 v14; // rax
  const void *v15; // rsi
  int v16; // eax
  unsigned int v17; // ecx
  __int64 v18; // r14
  void **p_s1; // rsi
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // r8
  int v23; // edx
  __int64 v24; // rax
  bool v25; // zf
  __int64 *v26; // rcx
  __int64 *v27; // r13
  _BYTE *v28; // r8
  size_t v29; // r15
  _BYTE *v30; // rdi
  _BYTE *v31; // r13
  __int64 v32; // rbx
  __int64 v33; // rax
  _QWORD **v34; // r15
  char *v35; // rdx
  char *v36; // r12
  unsigned __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // rax
  char *v40; // r12
  __int64 v41; // rax
  __int64 v42; // rax
  char *v43; // r13
  size_t v44; // [rsp+0h] [rbp-190h]
  int v45; // [rsp+Ch] [rbp-184h]
  size_t na; // [rsp+10h] [rbp-180h]
  unsigned int n; // [rsp+10h] [rbp-180h]
  char *nb; // [rsp+10h] [rbp-180h]
  __int64 v49; // [rsp+18h] [rbp-178h]
  __int64 v50; // [rsp+18h] [rbp-178h]
  _BYTE *v51; // [rsp+18h] [rbp-178h]
  void *s1; // [rsp+20h] [rbp-170h] BYREF
  size_t v53; // [rsp+28h] [rbp-168h]
  __int64 v54; // [rsp+30h] [rbp-160h] BYREF
  char *v55; // [rsp+40h] [rbp-150h] BYREF
  char *v56; // [rsp+48h] [rbp-148h]
  __int64 v57; // [rsp+50h] [rbp-140h] BYREF
  __int16 v58; // [rsp+60h] [rbp-130h]
  void *v59; // [rsp+70h] [rbp-120h] BYREF
  char *v60; // [rsp+78h] [rbp-118h]
  char v61; // [rsp+88h] [rbp-108h] BYREF
  __int64 *v62; // [rsp+108h] [rbp-88h]
  __int64 v63; // [rsp+118h] [rbp-78h] BYREF
  __int64 *v64; // [rsp+128h] [rbp-68h]
  __int64 v65; // [rsp+138h] [rbp-58h] BYREF
  char v66; // [rsp+150h] [rbp-40h]

  v4 = sub_97F930(**(_QWORD **)a1, **(_BYTE ***)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL), a2, a3);
  if ( !v4 )
    return;
  v5 = v4;
  if ( !*(_QWORD *)(v4 + 24) )
    return;
  sub_97F000((__int64 *)&s1, v4);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_DWORD *)(v6 + 24);
  if ( !v7 )
    goto LABEL_13;
  v8 = s1;
  v9 = v7 - 1;
  na = v53;
  v49 = *(_QWORD *)(v6 + 8);
  v10 = sub_C94890(s1, v53);
  v11 = na;
  v12 = 1;
  for ( i = v9 & v10; ; i = v9 & v17 )
  {
    v14 = v49 + 16LL * i;
    v15 = *(const void **)v14;
    if ( *(_QWORD *)v14 == -1 )
      break;
    if ( v15 == (const void *)-2LL )
    {
      if ( v8 == (void *)-2LL )
        goto LABEL_22;
    }
    else if ( v11 == *(_QWORD *)(v14 + 8) )
    {
      v45 = v12;
      n = i;
      if ( !v11 )
        goto LABEL_22;
      v44 = v11;
      v16 = memcmp(v8, v15, v11);
      v11 = v44;
      i = n;
      v12 = v45;
      if ( !v16 )
        goto LABEL_22;
    }
    v17 = v12 + i;
    ++v12;
  }
  if ( v8 != (void *)-1LL )
  {
LABEL_13:
    v18 = *(_QWORD *)(a1 + 24);
    p_s1 = &s1;
    v20 = *(unsigned int *)(v18 + 8);
    v21 = *(_QWORD *)v18;
    v22 = v20 + 1;
    v23 = *(_DWORD *)(v18 + 8);
    if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
    {
      if ( v21 > (unsigned __int64)&s1 || (unsigned __int64)&s1 >= v21 + 32 * v20 )
      {
        sub_95D880(*(_QWORD *)(a1 + 24), v22);
        v20 = *(unsigned int *)(v18 + 8);
        v21 = *(_QWORD *)v18;
        p_s1 = &s1;
        v23 = *(_DWORD *)(v18 + 8);
      }
      else
      {
        v43 = (char *)&s1 - v21;
        sub_95D880(*(_QWORD *)(a1 + 24), v22);
        v21 = *(_QWORD *)v18;
        v20 = *(unsigned int *)(v18 + 8);
        p_s1 = (void **)&v43[*(_QWORD *)v18];
        v23 = *(_DWORD *)(v18 + 8);
      }
    }
    v24 = 32 * v20;
    v25 = v24 + v21 == 0;
    v26 = (__int64 *)(v24 + v21);
    v27 = v26;
    if ( v25 )
      goto LABEL_21;
    *v26 = (__int64)(v26 + 2);
    v28 = *p_s1;
    v29 = (size_t)p_s1[1];
    if ( (char *)*p_s1 + v29 && !v28 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v59 = p_s1[1];
    if ( v29 > 0xF )
    {
      v51 = v28;
      v42 = sub_22409D0((__int64)v26, (unsigned __int64 *)&v59, 0);
      v28 = v51;
      *v27 = v42;
      v30 = (_BYTE *)v42;
      v27[2] = (__int64)v59;
    }
    else
    {
      v30 = (_BYTE *)*v26;
      if ( v29 == 1 )
      {
        *v30 = *v28;
        v29 = (size_t)v59;
        v30 = (_BYTE *)*v26;
LABEL_20:
        v27[1] = v29;
        v30[v29] = 0;
        v23 = *(_DWORD *)(v18 + 8);
LABEL_21:
        *(_DWORD *)(v18 + 8) = v23 + 1;
        goto LABEL_22;
      }
      if ( !v29 )
        goto LABEL_20;
    }
    memcpy(v30, v28, v29);
    v29 = (size_t)v59;
    v30 = (_BYTE *)*v27;
    goto LABEL_20;
  }
LABEL_22:
  v31 = sub_BA8CB0(**(_QWORD **)(a1 + 32), *(_QWORD *)(v5 + 16), *(_QWORD *)(v5 + 24));
  if ( !v31 )
  {
    v32 = *(_QWORD *)(a1 + 40);
    v33 = sub_B43CA0(v32);
    v34 = *(_QWORD ***)(v32 + 80);
    v50 = v33;
    sub_97F000((__int64 *)&v55, v5);
    sub_C0A940((__int64)&v59, v55, v56, (__int64)v34);
    if ( v55 != (char *)&v57 )
      j_j___libc_free_0((unsigned __int64)v55);
    v35 = *(char **)(v5 + 16);
    v36 = *(char **)(v5 + 24);
    nb = v35;
    v37 = sub_C0A1C0((int *)&v59, v34);
    v56 = v36;
    v38 = v37;
    v58 = 261;
    v55 = nb;
    v39 = sub_BD2DA0(136);
    v40 = (char *)v39;
    if ( v39 )
      sub_B2C3B0(v39, v38, 0, 0xFFFFFFFF, (__int64)&v55, v50);
    v41 = *(_QWORD *)(v32 - 32);
    if ( v41 )
    {
      if ( !*(_BYTE *)v41 && *(_QWORD *)(v41 + 24) == *(_QWORD *)(v32 + 80) )
        v31 = *(_BYTE **)(v32 - 32);
    }
    else
    {
      v31 = 0;
    }
    sub_B2EC90((__int64)v40, (__int64)v31);
    v55 = v40;
    sub_2A41DC0(v50, &v55, 1);
    if ( v66 )
    {
      v66 = 0;
      if ( v64 != &v65 )
        j_j___libc_free_0((unsigned __int64)v64);
      if ( v62 != &v63 )
        j_j___libc_free_0((unsigned __int64)v62);
      if ( v60 != &v61 )
        _libc_free((unsigned __int64)v60);
    }
  }
  if ( s1 != &v54 )
    j_j___libc_free_0((unsigned __int64)s1);
}
