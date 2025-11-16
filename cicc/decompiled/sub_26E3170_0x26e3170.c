// Function: sub_26E3170
// Address: 0x26e3170
//
void __fastcall sub_26E3170(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rdi
  void **v2; // rax
  const void **v3; // rbx
  const void *v4; // r15
  size_t v5; // r13
  int v6; // eax
  unsigned int v7; // r14d
  _QWORD *v8; // r8
  __int64 v9; // rax
  _QWORD *v10; // r8
  _QWORD *v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rbx
  __int128 v14; // rax
  __int64 v15; // rax
  size_t v16; // rdx
  size_t v17; // r12
  __int64 v18; // rdx
  __int128 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r14
  _QWORD *v22; // rax
  __int64 v23; // r14
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // r14
  int v29; // eax
  int v30; // r9d
  int v31; // ecx
  unsigned int i; // r8d
  __int64 v33; // rax
  const void *v34; // rsi
  unsigned int v35; // r8d
  unsigned __int64 v36; // r8
  __int64 v37; // r12
  __int64 v38; // rbx
  _QWORD *v39; // rdi
  unsigned __int64 v40; // rax
  _QWORD *v41; // rax
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // r9
  unsigned __int64 v44; // r10
  _QWORD *v45; // r11
  unsigned __int64 v46; // rsi
  _QWORD *v47; // rax
  _QWORD *v48; // r8
  int v49; // eax
  int v50; // [rsp+8h] [rbp-138h]
  unsigned int v51; // [rsp+Ch] [rbp-134h]
  __int64 v52; // [rsp+10h] [rbp-130h]
  __int64 v54; // [rsp+20h] [rbp-120h]
  int *v55; // [rsp+20h] [rbp-120h]
  unsigned int v56; // [rsp+20h] [rbp-120h]
  int v57; // [rsp+20h] [rbp-120h]
  _QWORD *v58; // [rsp+20h] [rbp-120h]
  int v59; // [rsp+20h] [rbp-120h]
  __int64 v60; // [rsp+28h] [rbp-118h]
  const void **s1; // [rsp+30h] [rbp-110h]
  _QWORD *s1a; // [rsp+30h] [rbp-110h]
  void *s1b; // [rsp+30h] [rbp-110h]
  _QWORD *v64; // [rsp+38h] [rbp-108h]
  __int64 v65; // [rsp+38h] [rbp-108h]
  _QWORD v66[2]; // [rsp+40h] [rbp-100h] BYREF
  unsigned __int64 v67; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v68; // [rsp+58h] [rbp-E8h]
  __int64 v69; // [rsp+60h] [rbp-E0h]
  __int64 v70[26]; // [rsp+70h] [rbp-D0h] BYREF

  v1 = a1[1];
  v67 = 0;
  v68 = 0;
  v69 = 0x800000000LL;
  v2 = (void **)(*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v1 + 48LL))(v1);
  if ( v2 )
  {
    v3 = (const void **)*v2;
    s1 = (const void **)v2[1];
    if ( *v2 != s1 )
    {
      while ( 1 )
      {
        v4 = *v3;
        v5 = (size_t)v3[1];
        if ( !*v3 )
          v5 = 0;
        v6 = sub_C92610();
        v7 = sub_C92740((__int64)&v67, v4, v5, v6);
        v8 = (_QWORD *)(v67 + 8LL * v7);
        if ( !*v8 )
          goto LABEL_10;
        if ( *v8 == -8 )
        {
          LODWORD(v69) = v69 - 1;
LABEL_10:
          v64 = (_QWORD *)(v67 + 8LL * v7);
          v9 = sub_C7D670(v5 + 9, 8);
          v10 = v64;
          v11 = (_QWORD *)v9;
          if ( v5 )
          {
            v58 = (_QWORD *)v9;
            memcpy((void *)(v9 + 8), v4, v5);
            v10 = v64;
            v11 = v58;
          }
          *((_BYTE *)v11 + v5 + 8) = 0;
          v3 += 2;
          *v11 = v5;
          *v10 = v11;
          ++HIDWORD(v68);
          sub_C929D0((__int64 *)&v67, v7);
          if ( s1 == v3 )
            break;
        }
        else
        {
          v3 += 2;
          if ( s1 == v3 )
            break;
        }
      }
    }
  }
  v12 = *(_QWORD *)(*a1 + 32);
  v65 = *a1 + 24;
  while ( v65 != v12 )
  {
    v13 = v12 - 56;
    if ( !v12 )
      v13 = 0;
    if ( sub_B2FC80(v13) )
      goto LABEL_16;
    *(_QWORD *)&v14 = sub_BD5D20(v13);
    v15 = sub_C16140(v14, (__int64)"selected", 8);
    v52 = v16;
    v17 = v16;
    s1a = (_QWORD *)v15;
    v70[0] = sub_B2D7E0(v13, "sample-profile-suffix-elision-policy", 0x24u);
    v54 = sub_A72240(v70);
    v60 = v18;
    *(_QWORD *)&v19 = sub_BD5D20(v13);
    v55 = (int *)sub_C16140(v19, v54, v60);
    v21 = v20;
    if ( v55 )
    {
      sub_C7D030(v70);
      sub_C7D280((int *)v70, v55, v21);
      sub_C7D290(v70, v66);
      v21 = v66[0];
    }
    v70[0] = v21;
    v22 = sub_C1DD00(a1 + 5, v21 % a1[6], v70, v21);
    if ( v22 )
    {
      if ( *v22 )
        goto LABEL_16;
    }
    v23 = v67;
    v56 = v68;
    v24 = sub_C92610();
    v25 = sub_C92860((__int64 *)&v67, s1a, v17, v24);
    v26 = v25 == -1 ? v67 + 8LL * (unsigned int)v68 : v67 + 8LL * v25;
    if ( v26 != v23 + 8LL * v56 )
      goto LABEL_16;
    v27 = a1[41];
    if ( v27 )
    {
      v57 = *(_DWORD *)(v27 + 32);
      if ( v57 )
      {
        v28 = *(_QWORD *)(v27 + 16);
        v29 = sub_C94890(s1a, v17);
        v30 = 1;
        v31 = v57 - 1;
        for ( i = (v57 - 1) & v29; ; i = v31 & v35 )
        {
          v33 = v28 + 16LL * i;
          v34 = *(const void **)v33;
          if ( *(_QWORD *)v33 == -1 )
            break;
          if ( v34 == (const void *)-2LL )
          {
            if ( s1a == (_QWORD *)-2LL )
              goto LABEL_16;
          }
          else if ( *(_QWORD *)(v33 + 8) == v17 )
          {
            v50 = v30;
            v51 = i;
            v59 = v31;
            if ( !v17 )
              goto LABEL_16;
            v49 = memcmp(s1a, v34, v17);
            v31 = v59;
            i = v51;
            v30 = v50;
            if ( !v49 )
              goto LABEL_16;
          }
          v35 = v30 + i;
          ++v30;
        }
        if ( s1a == (_QWORD *)-1LL )
          goto LABEL_16;
      }
    }
    if ( s1a )
    {
      sub_C7D030(v70);
      sub_C7D280((int *)v70, (int *)s1a, v17);
      sub_C7D290(v70, v66);
      v52 = v66[0];
    }
    v70[0] = v52;
    v40 = (unsigned __int64)sub_26C56D0(a1 + 34, v70);
    if ( !v40 )
    {
      v41 = (_QWORD *)sub_22077B0(0x18u);
      v42 = (unsigned __int64)v41;
      if ( v41 )
        *v41 = 0;
      v43 = v70[0];
      v41[2] = 0;
      v44 = a1[35];
      v41[1] = v43;
      v45 = *(_QWORD **)(a1[34] + 8 * (v43 % v44));
      v46 = v43 % v44;
      if ( !v45 )
        goto LABEL_61;
      v47 = (_QWORD *)*v45;
      if ( *(_QWORD *)(*v45 + 8LL) != v43 )
      {
        while ( 1 )
        {
          v48 = (_QWORD *)*v47;
          if ( !*v47 )
            break;
          v45 = v47;
          if ( v46 != v48[1] % v44 )
            break;
          v47 = (_QWORD *)*v47;
          if ( v48[1] == v43 )
            goto LABEL_56;
        }
LABEL_61:
        v40 = sub_26DFD20(a1 + 34, v46, v43, v42, 1);
        goto LABEL_45;
      }
LABEL_56:
      if ( !*v45 )
        goto LABEL_61;
      s1b = (void *)*v45;
      j_j___libc_free_0(v42);
      v40 = (unsigned __int64)s1b;
    }
LABEL_45:
    *(_QWORD *)(v40 + 16) = v13;
LABEL_16:
    v12 = *(_QWORD *)(v12 + 8);
  }
  v36 = v67;
  if ( HIDWORD(v68) && (_DWORD)v68 )
  {
    v37 = 8LL * (unsigned int)v68;
    v38 = 0;
    do
    {
      v39 = *(_QWORD **)(v36 + v38);
      if ( v39 != (_QWORD *)-8LL && v39 )
      {
        sub_C7D6A0((__int64)v39, *v39 + 9LL, 8);
        v36 = v67;
      }
      v38 += 8;
    }
    while ( v38 != v37 );
  }
  _libc_free(v36);
}
