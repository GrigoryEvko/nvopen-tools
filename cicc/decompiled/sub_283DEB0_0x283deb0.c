// Function: sub_283DEB0
// Address: 0x283deb0
//
__int64 __fastcall sub_283DEB0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 *v8; // r15
  __int64 *v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  _QWORD *v12; // rdx
  _QWORD *v13; // rbx
  __int64 v14; // r12
  _QWORD *v15; // r14
  _QWORD *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rsi
  void (__fastcall *v19)(_QWORD *, __int64, __int64, char *); // r8
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  void **v23; // rax
  unsigned __int64 v24; // rdx
  __int64 *v25; // rax
  bool v26; // zf
  __int64 v27; // r13
  _QWORD *v28; // rbx
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // rdi
  void **v32; // rdx
  unsigned __int64 v33; // rcx
  __int64 **v34; // rax
  _QWORD *v35; // rbx
  __int64 v36; // r14
  _QWORD *v37; // rax
  _QWORD *v38; // rdi
  _QWORD *v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rsi
  unsigned __int64 v42; // rdi
  __int64 v44; // [rsp-8h] [rbp-218h]
  __int64 v45; // [rsp+10h] [rbp-200h]
  __int64 v47; // [rsp+20h] [rbp-1F0h]
  _QWORD *v48; // [rsp+28h] [rbp-1E8h]
  int v50; // [rsp+40h] [rbp-1D0h]
  unsigned int v51; // [rsp+44h] [rbp-1CCh]
  unsigned __int64 v54; // [rsp+58h] [rbp-1B8h]
  __int64 v55; // [rsp+60h] [rbp-1B0h]
  char v56; // [rsp+70h] [rbp-1A0h]
  char v57; // [rsp+76h] [rbp-19Ah]
  char v58; // [rsp+77h] [rbp-199h]
  __int64 v61; // [rsp+88h] [rbp-188h]
  __int64 v62; // [rsp+98h] [rbp-178h] BYREF
  char v63[8]; // [rsp+A0h] [rbp-170h] BYREF
  unsigned __int64 v64; // [rsp+A8h] [rbp-168h]
  char v65; // [rsp+BCh] [rbp-154h]
  char v66[16]; // [rsp+C0h] [rbp-150h] BYREF
  char v67[8]; // [rsp+D0h] [rbp-140h] BYREF
  unsigned __int64 v68; // [rsp+D8h] [rbp-138h]
  char v69; // [rsp+ECh] [rbp-124h]
  char v70[16]; // [rsp+F0h] [rbp-120h] BYREF
  unsigned __int64 v71[14]; // [rsp+100h] [rbp-110h] BYREF
  _QWORD *v72; // [rsp+170h] [rbp-A0h] BYREF
  unsigned __int64 v73; // [rsp+178h] [rbp-98h]
  char v74; // [rsp+18Ch] [rbp-84h]
  _BYTE v75[16]; // [rsp+190h] [rbp-80h] BYREF
  char v76[8]; // [rsp+1A0h] [rbp-70h] BYREF
  unsigned __int64 v77; // [rsp+1A8h] [rbp-68h]
  char v78; // [rsp+1BCh] [rbp-54h]
  _BYTE v79[16]; // [rsp+1C0h] [rbp-50h] BYREF
  char v80; // [rsp+1D0h] [rbp-40h]

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  v62 = *(_QWORD *)(sub_22D3D20(a4, &qword_4F8A320, a3, a5) + 8);
  v47 = *(unsigned int *)(a2 + 64);
  if ( !*(_DWORD *)(a2 + 64) )
    return a1;
  v7 = 0;
  v8 = a3;
  v54 = 0;
  v61 = 0;
  v58 = 0;
  v50 = 0;
  while ( 1 )
  {
    memset(v71, 0, 0x68u);
    v55 = (unsigned int)v61 >> 6;
    v56 = v61 & 0x3F;
    v57 = (*(_QWORD *)(*(_QWORD *)a2 + 8 * v55) >> v61) & 1;
    if ( ((*(_QWORD *)(*(_QWORD *)a2 + 8 * v55) >> v61) & 1) == 0 )
    {
      sub_283EB30((unsigned int)&v72, (_DWORD)a3, *(_QWORD *)(a2 + 72) + 8 * v50, a4, a5, a6, (__int64)&v62);
      v10 = v44;
      if ( LOBYTE(v71[12]) )
      {
        if ( v80 )
        {
          sub_C8CF80((__int64)v71, &v71[4], 2, (__int64)v75, (__int64)&v72);
          sub_C8CF80((__int64)&v71[6], &v71[10], 2, (__int64)v79, (__int64)v76);
        }
        else
        {
          LOBYTE(v71[12]) = 0;
          if ( !BYTE4(v71[9]) )
            _libc_free(v71[7]);
          if ( !BYTE4(v71[3]) )
            _libc_free(v71[1]);
        }
      }
      else
      {
        if ( !v80 )
        {
          ++v50;
          goto LABEL_5;
        }
        sub_C8CF70((__int64)v71, &v71[4], 2, (__int64)v75, (__int64)&v72);
        sub_C8CF70((__int64)&v71[6], &v71[10], 2, (__int64)v79, (__int64)v76);
        LOBYTE(v71[12]) = 1;
      }
      if ( !v80 )
        goto LABEL_61;
      if ( v78 )
      {
        if ( v74 )
          goto LABEL_61;
      }
      else
      {
        _libc_free(v77);
        if ( v74 )
          goto LABEL_61;
      }
      _libc_free(v73);
LABEL_61:
      ++v50;
      v57 = v58;
      if ( LOBYTE(v71[12]) )
        goto LABEL_28;
      goto LABEL_62;
    }
    v51 = v7 + 1;
    v9 = (__int64 *)(*(_QWORD *)(a2 + 96) + 8 * v7);
    if ( !v58 || *(_BYTE *)(a6 + 26) )
    {
      do
      {
        v27 = (__int64)v8;
        v8 = (__int64 *)*v8;
      }
      while ( v8 );
      sub_D56E90((__int64 *)&v72, v27, *(_QWORD *)(a5 + 32));
      v28 = v72;
      v72 = 0;
      if ( v54 )
      {
        v29 = *(_QWORD *)(v54 + 8);
        if ( v29 != v54 + 24 )
          _libc_free(v29);
        j_j___libc_free_0(v54);
        v30 = (unsigned __int64)v72;
        if ( v72 )
        {
          v31 = v72[1];
          if ( (_QWORD *)v31 != v72 + 3 )
            _libc_free(v31);
          j_j___libc_free_0(v30);
        }
      }
      v54 = (unsigned __int64)v28;
      v8 = (__int64 *)v27;
      *(_BYTE *)(a6 + 26) = 0;
    }
    v45 = **(_QWORD **)(v54 + 8);
    v58 = sub_283DC30(&v62, *v9, v45);
    if ( v58 )
      break;
    v80 = 0;
    if ( LOBYTE(v71[12]) )
      goto LABEL_63;
    v7 = v51;
    v58 = v57;
LABEL_5:
    if ( v47 == ++v61 )
      goto LABEL_94;
  }
  (*(void (__fastcall **)(char *, __int64, unsigned __int64, __int64, __int64, __int64))(*(_QWORD *)*v9 + 16LL))(
    v63,
    *v9,
    v54,
    a4,
    a5,
    a6);
  if ( *(_BYTE *)(a6 + 24) )
  {
    if ( v62 )
    {
      v12 = *(_QWORD **)(v62 + 576);
      v13 = &v12[4 * *(unsigned int *)(v62 + 584)];
      if ( v12 != v13 )
      {
        v14 = *v9;
        v15 = *(_QWORD **)(v62 + 576);
        do
        {
          v16 = v15;
          v18 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 32LL))(v14);
          v19 = *(void (__fastcall **)(_QWORD *, __int64, __int64, char *))(v15[3] & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v15[3] & 2) == 0 )
            v16 = (_QWORD *)*v15;
          v15 += 4;
          v19(v16, v18, v17, v63);
        }
        while ( v13 != v15 );
      }
    }
  }
  else if ( v62 )
  {
    v35 = *(_QWORD **)(v62 + 432);
    v48 = &v35[4 * *(unsigned int *)(v62 + 440)];
    if ( v35 != v48 )
    {
      v36 = *v9;
      do
      {
        v72 = 0;
        v37 = (_QWORD *)sub_22077B0(0x10u);
        if ( v37 )
        {
          v37[1] = v45;
          *v37 = &unk_4A09EA8;
        }
        v38 = v72;
        v72 = v37;
        if ( v38 )
          (*(void (__fastcall **)(_QWORD *))(*v38 + 8LL))(v38);
        v39 = v35;
        v41 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v36 + 32LL))(v36);
        if ( (v35[3] & 2) == 0 )
          v39 = (_QWORD *)*v35;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, char *))(v35[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v39,
          v41,
          v40,
          &v72,
          v63);
        if ( v72 )
          (*(void (__fastcall **)(_QWORD *))(*v72 + 8LL))(v72);
        v35 += 4;
      }
      while ( v48 != v35 );
    }
  }
  sub_C8CF70((__int64)&v72, v75, 2, (__int64)v66, (__int64)v63);
  sub_C8CF70((__int64)v76, v79, 2, (__int64)v70, (__int64)v67);
  v80 = 1;
  if ( v69 )
  {
    if ( v65 )
      goto LABEL_19;
LABEL_92:
    _libc_free(v64);
  }
  else
  {
    _libc_free(v68);
    if ( !v65 )
      goto LABEL_92;
  }
LABEL_19:
  if ( LOBYTE(v71[12]) )
  {
    if ( v80 )
    {
      sub_C8CF80((__int64)v71, &v71[4], 2, (__int64)v75, (__int64)&v72);
      sub_C8CF80((__int64)&v71[6], &v71[10], 2, (__int64)v79, (__int64)v76);
    }
    else
    {
LABEL_63:
      LOBYTE(v71[12]) = 0;
      if ( !BYTE4(v71[9]) )
        _libc_free(v71[7]);
      if ( !BYTE4(v71[3]) )
        _libc_free(v71[1]);
    }
  }
  else
  {
    if ( !v80 )
    {
      v7 = v51;
      goto LABEL_5;
    }
    sub_C8CF70((__int64)v71, &v71[4], 2, (__int64)v75, (__int64)&v72);
    sub_C8CF70((__int64)&v71[6], &v71[10], 2, (__int64)v79, (__int64)v76);
    LOBYTE(v71[12]) = 1;
  }
  if ( v80 )
  {
    if ( !v78 )
      _libc_free(v77);
    if ( !v74 )
      _libc_free(v73);
  }
  v7 = v51;
  if ( !LOBYTE(v71[12]) )
  {
LABEL_62:
    v58 = v57;
    goto LABEL_5;
  }
LABEL_28:
  v58 = *(_BYTE *)(a6 + 24);
  if ( !v58 )
  {
    v20 = (__int64)a3;
    if ( ((*(_QWORD *)(*(_QWORD *)a2 + 8 * v55) >> v56) & 1) != 0 )
      v20 = (__int64)v8;
    sub_22D08B0(a4, v20, (__int64)v71);
    sub_BBADB0(a1, (__int64)v71, v21, v22);
    if ( BYTE4(v71[9]) )
    {
      v23 = (void **)v71[7];
      v24 = v71[7] + 8LL * HIDWORD(v71[8]);
      if ( v71[7] != v24 )
      {
        while ( *v23 != &unk_4F876D0 )
        {
          if ( (void **)v24 == ++v23 )
            goto LABEL_68;
        }
        goto LABEL_36;
      }
LABEL_68:
      if ( BYTE4(v71[3]) )
      {
        v32 = (void **)v71[1];
        v33 = v71[1] + 8LL * HIDWORD(v71[2]);
        if ( v71[1] != v33 )
        {
          v34 = (__int64 **)v71[1];
          while ( *v34 != &qword_4F82400 )
          {
            if ( (__int64 **)v33 == ++v34 )
              goto LABEL_108;
          }
LABEL_73:
          v58 = v57;
        }
      }
      else
      {
        if ( sub_C8CA60((__int64)v71, (__int64)&qword_4F82400) )
          goto LABEL_73;
        if ( BYTE4(v71[3]) )
        {
          v32 = (void **)v71[1];
          v34 = (__int64 **)(v71[1] + 8LL * HIDWORD(v71[2]));
          if ( v34 != (__int64 **)v71[1] )
          {
LABEL_108:
            while ( *v32 != &unk_4F876D0 )
            {
              if ( v34 == (__int64 **)++v32 )
                goto LABEL_36;
            }
            v58 = v57;
          }
        }
        else
        {
          v58 = v57 & (sub_C8CA60((__int64)v71, (__int64)&unk_4F876D0) != 0);
        }
      }
    }
    else if ( !sub_C8CA60((__int64)&v71[6], (__int64)&unk_4F876D0) )
    {
      goto LABEL_68;
    }
LABEL_36:
    v25 = a3;
    if ( ((*(_QWORD *)(*(_QWORD *)a2 + 8 * v55) >> v56) & 1) != 0 )
      v25 = v8;
    v26 = LOBYTE(v71[12]) == 0;
    *(_QWORD *)(a6 + 32) = *v25;
    if ( !v26 )
    {
      if ( !BYTE4(v71[9]) )
        _libc_free(v71[7]);
      if ( !BYTE4(v71[3]) )
        _libc_free(v71[1]);
    }
    goto LABEL_5;
  }
  sub_BBADB0(a1, (__int64)v71, v10, v11);
  if ( LOBYTE(v71[12]) )
  {
    if ( !BYTE4(v71[9]) )
      _libc_free(v71[7]);
    if ( !BYTE4(v71[3]) )
      _libc_free(v71[1]);
  }
LABEL_94:
  if ( v54 )
  {
    v42 = *(_QWORD *)(v54 + 8);
    if ( v42 != v54 + 24 )
      _libc_free(v42);
    j_j___libc_free_0(v54);
  }
  return a1;
}
