// Function: sub_39422E0
// Address: 0x39422e0
//
__int64 __fastcall sub_39422E0(_QWORD *a1, unsigned __int64 *a2)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // r8
  __int64 v8; // r13
  unsigned int v9; // r13d
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  void *v17; // r14
  unsigned __int64 v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // r13
  _QWORD *v25; // r12
  size_t v26; // r14
  __int64 v27; // r15
  size_t v28; // rbx
  size_t v29; // rdx
  int v30; // eax
  __int64 v31; // rbx
  size_t v32; // r15
  size_t v33; // rbx
  size_t v34; // rdx
  signed __int64 v35; // rax
  unsigned __int64 v36; // rcx
  __int64 v37; // rax
  size_t v38; // rax
  void *v39; // rdx
  _QWORD *v40; // rdi
  __int64 v41; // [rsp-10h] [rbp-1A0h]
  unsigned __int64 v42; // [rsp+18h] [rbp-178h]
  __int64 v43; // [rsp+20h] [rbp-170h]
  unsigned __int64 *v44; // [rsp+28h] [rbp-168h]
  unsigned int v45; // [rsp+30h] [rbp-160h]
  __int64 v46; // [rsp+30h] [rbp-160h]
  int v47; // [rsp+40h] [rbp-150h]
  _QWORD *v48; // [rsp+48h] [rbp-148h]
  void **p_s2; // [rsp+58h] [rbp-138h] BYREF
  unsigned __int64 v50; // [rsp+60h] [rbp-130h] BYREF
  char v51; // [rsp+70h] [rbp-120h]
  unsigned int v52; // [rsp+80h] [rbp-110h] BYREF
  char v53; // [rsp+90h] [rbp-100h]
  size_t v54; // [rsp+A0h] [rbp-F0h] BYREF
  char v55; // [rsp+B0h] [rbp-E0h]
  _QWORD v56[2]; // [rsp+C0h] [rbp-D0h] BYREF
  char v57; // [rsp+D0h] [rbp-C0h]
  unsigned __int64 v58; // [rsp+E0h] [rbp-B0h] BYREF
  char v59; // [rsp+F0h] [rbp-A0h]
  __int64 v60; // [rsp+100h] [rbp-90h] BYREF
  char v61; // [rsp+110h] [rbp-80h]
  void *src; // [rsp+120h] [rbp-70h] BYREF
  size_t n; // [rsp+128h] [rbp-68h]
  char v64; // [rsp+130h] [rbp-60h]
  void *s2; // [rsp+140h] [rbp-50h] BYREF
  unsigned __int64 v66; // [rsp+148h] [rbp-48h]
  _QWORD v67[8]; // [rsp+150h] [rbp-40h] BYREF

  sub_393FF90((__int64)&v50, a1);
  if ( (v51 & 1) != 0 )
  {
    result = (unsigned int)v50;
    if ( (_DWORD)v50 )
      return result;
  }
  a2[2] = sub_393FEE0(v50, 1u, a2[2], (bool *)&s2);
  sub_3940120((__int64)&v52, a1);
  if ( (v53 & 1) != 0 )
  {
    result = v52;
    if ( v52 )
      return result;
  }
  if ( !v52 )
  {
LABEL_40:
    v11 = v56;
    v12 = (__int64)a1;
    sub_3940120((__int64)v56, a1);
    if ( (v57 & 1) != 0 )
    {
      result = LODWORD(v56[0]);
      v13 = v56[1];
      if ( LODWORD(v56[0]) )
        return result;
    }
    if ( !LODWORD(v56[0]) )
      goto LABEL_96;
    v48 = a1;
    v47 = 0;
    v44 = a2;
LABEL_44:
    sub_393FF90((__int64)&v58, v48);
    if ( (v59 & 1) != 0 )
    {
      result = (unsigned int)v58;
      if ( (_DWORD)v58 )
        return result;
    }
    sub_393FF90((__int64)&v60, v48);
    if ( (v61 & 1) != 0 )
    {
      result = (unsigned int)v60;
      if ( (_DWORD)v60 )
        return result;
    }
    (*(void (__fastcall **)(void **))(*v48 + 48LL))(&src);
    if ( (v64 & 1) != 0 )
    {
      result = (unsigned int)src;
      if ( (_DWORD)src )
        return result;
    }
    v17 = src;
    if ( !src )
    {
      v66 = 0;
      s2 = v67;
      LOBYTE(v67[0]) = 0;
LABEL_55:
      v20 = v44[12];
      v21 = (unsigned int)v60;
      v54 = __PAIR64__(v60, v58);
      v22 = (__int64)(v44 + 11);
      if ( !v20 )
        goto LABEL_98;
      do
      {
        if ( (unsigned int)v58 > *(_DWORD *)(v20 + 32)
          || (_DWORD)v58 == *(_DWORD *)(v20 + 32) && (unsigned int)v60 > *(_DWORD *)(v20 + 36) )
        {
          v20 = *(_QWORD *)(v20 + 24);
        }
        else
        {
          v22 = v20;
          v20 = *(_QWORD *)(v20 + 16);
        }
      }
      while ( v20 );
      if ( (unsigned __int64 *)v22 == v44 + 11
        || (unsigned int)v58 < *(_DWORD *)(v22 + 32)
        || (_DWORD)v58 == *(_DWORD *)(v22 + 32) && (unsigned int)v60 < *(_DWORD *)(v22 + 36) )
      {
LABEL_98:
        p_s2 = (void **)&v54;
        v22 = sub_3941C40(v44 + 10, v22, (__int64 **)&p_s2);
      }
      v23 = *(_QWORD *)(v22 + 56);
      v46 = v22 + 48;
      if ( !v23 )
      {
        v24 = v22 + 48;
        goto LABEL_91;
      }
      v43 = v22;
      v24 = v22 + 48;
      v25 = s2;
      v26 = v66;
      v27 = v23;
      while ( 1 )
      {
        v28 = *(_QWORD *)(v27 + 40);
        v29 = v26;
        if ( v28 <= v26 )
          v29 = *(_QWORD *)(v27 + 40);
        if ( v29 )
        {
          v30 = memcmp(*(const void **)(v27 + 32), v25, v29);
          if ( v30 )
            goto LABEL_81;
        }
        v31 = v28 - v26;
        if ( v31 >= 0x80000000LL )
          goto LABEL_82;
        if ( v31 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          break;
LABEL_72:
        v27 = *(_QWORD *)(v27 + 24);
LABEL_73:
        if ( !v27 )
        {
          v32 = v26;
          v22 = v43;
          if ( v46 == v24 )
            goto LABEL_91;
          v33 = *(_QWORD *)(v24 + 40);
          v34 = v32;
          if ( v33 <= v32 )
            v34 = *(_QWORD *)(v24 + 40);
          if ( v34 && (LODWORD(v35) = memcmp(v25, *(const void **)(v24 + 32), v34), (_DWORD)v35) )
          {
LABEL_90:
            if ( (int)v35 < 0 )
              goto LABEL_91;
          }
          else
          {
            v36 = 0x80000000LL;
            v35 = v32 - v33;
            if ( (__int64)(v32 - v33) < 0x80000000LL )
            {
              v36 = 0xFFFFFFFF7FFFFFFFLL;
              if ( v35 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                goto LABEL_90;
LABEL_91:
              p_s2 = &s2;
              v37 = sub_3942120((_QWORD *)(v22 + 40), (_QWORD *)v24, (__m128i **)&p_s2);
              v25 = s2;
              v24 = v37;
            }
          }
          if ( v25 != v67 )
            j_j___libc_free_0((unsigned __int64)v25);
          v38 = n;
          v39 = src;
          v12 = v24 + 64;
          v11 = v48;
          *(_QWORD *)(v24 + 64) = src;
          *(_QWORD *)(v24 + 72) = v38;
          result = sub_39422E0(v48, v24 + 64, v39, v36, v21);
          if ( (_DWORD)result )
            return result;
          if ( LODWORD(v56[0]) <= ++v47 )
          {
LABEL_96:
            sub_393D180((__int64)v11, v12, v13, v14, v15, v16);
            return 0;
          }
          goto LABEL_44;
        }
      }
      v30 = v31;
LABEL_81:
      if ( v30 >= 0 )
      {
LABEL_82:
        v24 = v27;
        v27 = *(_QWORD *)(v27 + 16);
        goto LABEL_73;
      }
      goto LABEL_72;
    }
    v18 = n;
    s2 = v67;
    v54 = n;
    if ( n > 0xF )
    {
      s2 = (void *)sub_22409D0((__int64)&s2, &v54, 0);
      v40 = s2;
      v67[0] = v54;
    }
    else
    {
      if ( n == 1 )
      {
        LOBYTE(v67[0]) = *(_BYTE *)src;
        v19 = v67;
LABEL_54:
        v66 = v18;
        *((_BYTE *)v19 + v18) = 0;
        goto LABEL_55;
      }
      if ( !n )
      {
        v19 = v67;
        goto LABEL_54;
      }
      v40 = v67;
    }
    memcpy(v40, v17, v18);
    v18 = v54;
    v19 = s2;
    goto LABEL_54;
  }
  v45 = 0;
  while ( 1 )
  {
    sub_393FF90((__int64)&v54, a1);
    if ( (v55 & 1) != 0 )
    {
      result = (unsigned int)v54;
      if ( (_DWORD)v54 )
        return result;
    }
    if ( (v54 & 0xFFFF0000) != 0 )
    {
      sub_2241E40();
      return 0;
    }
    sub_393FF90((__int64)v56, a1);
    if ( (v57 & 1) != 0 )
    {
      result = LODWORD(v56[0]);
      if ( LODWORD(v56[0]) )
        return result;
    }
    sub_393FF90((__int64)&v58, a1);
    if ( (v59 & 1) != 0 )
    {
      result = (unsigned int)v58;
      if ( (_DWORD)v58 )
        return result;
    }
    sub_3940120((__int64)&v60, a1);
    if ( (v61 & 1) != 0 )
    {
      result = (unsigned int)v60;
      if ( (_DWORD)v60 )
        return result;
    }
    else if ( (_DWORD)v60 )
    {
      v9 = 0;
      do
      {
        (*(void (__fastcall **)(void **, _QWORD *, __int64))(*a1 + 48LL))(&src, a1, v5);
        if ( (v64 & 1) != 0 )
        {
          result = (unsigned int)src;
          if ( (_DWORD)src )
            return result;
        }
        sub_393FF90((__int64)&s2, a1);
        if ( (v67[0] & 1) != 0 )
        {
          result = (unsigned int)s2;
          if ( (_DWORD)s2 )
            return result;
        }
        ++v9;
        sub_39417C0((__int64)a2, v54, v56[0], (unsigned __int8 *)src, n, (unsigned __int64)s2, 1u);
        v5 = v41;
      }
      while ( (unsigned int)v60 > v9 );
    }
    v6 = a2[6];
    v7 = v58;
    s2 = (void *)__PAIR64__(v56[0], v54);
    v8 = (__int64)(a2 + 5);
    if ( !v6 )
      goto LABEL_38;
    do
    {
      if ( (unsigned int)v54 > *(_DWORD *)(v6 + 32)
        || (_DWORD)v54 == *(_DWORD *)(v6 + 32) && LODWORD(v56[0]) > *(_DWORD *)(v6 + 36) )
      {
        v6 = *(_QWORD *)(v6 + 24);
      }
      else
      {
        v8 = v6;
        v6 = *(_QWORD *)(v6 + 16);
      }
    }
    while ( v6 );
    if ( a2 + 5 == (unsigned __int64 *)v8
      || (unsigned int)v54 < *(_DWORD *)(v8 + 32)
      || (_DWORD)v54 == *(_DWORD *)(v8 + 32) && LODWORD(v56[0]) < *(_DWORD *)(v8 + 36) )
    {
LABEL_38:
      src = &s2;
      v42 = v58;
      v10 = sub_39416E0(a2 + 4, v8, (__int64 **)&src);
      v7 = v42;
      v8 = v10;
    }
    ++v45;
    *(_QWORD *)(v8 + 40) = sub_393FEE0(v7, 1u, *(_QWORD *)(v8 + 40), (bool *)&src);
    if ( v52 <= v45 )
      goto LABEL_40;
  }
}
