// Function: sub_2DEBFC0
// Address: 0x2debfc0
//
void __fastcall sub_2DEBFC0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v7; // al
  int v8; // r13d
  _BYTE *v9; // r14
  _BYTE *v10; // r15
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r13d
  unsigned int v14; // ebx
  const void **v15; // rsi
  _QWORD *v16; // rax
  unsigned int v17; // r15d
  unsigned int v18; // eax
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // eax
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  _BYTE *v27; // rax
  unsigned __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned int v31; // eax
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // rsi
  int v39; // edx
  __int64 v40; // rcx
  unsigned __int64 *v41; // r14
  __int64 v42; // rax
  int v43; // edx
  bool v44; // cc
  bool v47; // zf
  unsigned __int64 v48; // rbx
  __int64 v49; // rdi
  __int64 v50; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v51; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v52; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v53; // [rsp+28h] [rbp-B8h] BYREF
  _BYTE *v54; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+38h] [rbp-A8h]
  _BYTE v56[96]; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 v57; // [rsp+A0h] [rbp-40h]
  unsigned int v58; // [rsp+A8h] [rbp-38h]

  v7 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x1Cu || (v8 = v7, (unsigned int)v7 - 42 > 0x11) )
  {
    LODWORD(v52) = -1;
    v20 = 0xFFFFFFFFLL;
    v55 = 0x400000000LL;
    v21 = *(_QWORD *)(a1 + 8);
    v53 = a1;
    v54 = v56;
    v58 = 1;
    v57 = 0;
    if ( *(_BYTE *)(v21 + 8) == 12 )
    {
      LODWORD(v52) = 0;
      v20 = 0;
      a4 = 0;
      v22 = *(_DWORD *)(v21 + 8) >> 8;
      v51 = v22;
      if ( v22 > 0x40 )
      {
        sub_C43690((__int64)&v50, 0, 0);
        if ( v58 > 0x40 && v57 )
          j_j___libc_free_0_0(v57);
        a4 = v50;
        v22 = v51;
        v20 = (unsigned int)v52;
        a1 = v53;
      }
      v57 = a4;
      v58 = v22;
    }
    *(_QWORD *)(a2 + 8) = a1;
    *(_DWORD *)a2 = v20;
    sub_2DEB400(a2 + 16, (unsigned __int64 *)&v54, v20, a4, a5, a6);
    if ( *(_DWORD *)(a2 + 136) > 0x40u )
    {
      v26 = *(_QWORD *)(a2 + 128);
      if ( v26 )
        j_j___libc_free_0_0(v26);
    }
    v23 = (unsigned __int64)v54;
    *(_QWORD *)(a2 + 128) = v57;
    *(_DWORD *)(a2 + 136) = v58;
    v24 = v23 + 24LL * (unsigned int)v55;
    if ( v23 == v24 )
      goto LABEL_35;
    do
    {
      v24 -= 24LL;
      if ( *(_DWORD *)(v24 + 16) > 0x40u )
      {
        v25 = *(_QWORD *)(v24 + 8);
        if ( v25 )
          j_j___libc_free_0_0(v25);
      }
    }
    while ( v23 != v24 );
    goto LABEL_34;
  }
  v9 = *(_BYTE **)(a1 - 32);
  v10 = *(_BYTE **)(a1 - 64);
  if ( *v9 != 17 )
  {
    if ( !sub_B46D50((unsigned __int8 *)a1) || *v10 != 17 )
    {
LABEL_46:
      v28 = a1;
      v29 = 0xFFFFFFFFLL;
      v55 = 0x400000000LL;
      v30 = *(_QWORD *)(a1 + 8);
      LODWORD(v52) = -1;
      v53 = a1;
      v54 = v56;
      v58 = 1;
      v57 = 0;
      if ( *(_BYTE *)(v30 + 8) == 12 )
      {
        LODWORD(v52) = 0;
        v29 = 0;
        a4 = 0;
        v31 = *(_DWORD *)(v30 + 8) >> 8;
        v51 = v31;
        if ( v31 > 0x40 )
        {
          sub_C43690((__int64)&v50, 0, 0);
          if ( v58 > 0x40 && v57 )
            j_j___libc_free_0_0(v57);
          a4 = v50;
          v31 = v51;
          v29 = (unsigned int)v52;
          v28 = v53;
        }
        v57 = a4;
        v58 = v31;
      }
      *(_DWORD *)a2 = v29;
      *(_QWORD *)(a2 + 8) = v28;
      sub_2DEB400(a2 + 16, (unsigned __int64 *)&v54, v29, a4, a5, a6);
      if ( *(_DWORD *)(a2 + 136) > 0x40u )
      {
        v32 = *(_QWORD *)(a2 + 128);
        if ( v32 )
          j_j___libc_free_0_0(v32);
      }
      v33 = (unsigned __int64)v54;
      *(_QWORD *)(a2 + 128) = v57;
      *(_DWORD *)(a2 + 136) = v58;
      v24 = v33 + 24LL * (unsigned int)v55;
      if ( v33 == v24 )
        goto LABEL_35;
      do
      {
        v24 -= 24LL;
        if ( *(_DWORD *)(v24 + 16) > 0x40u )
        {
          v34 = *(_QWORD *)(v24 + 8);
          if ( v34 )
            j_j___libc_free_0_0(v34);
        }
      }
      while ( v33 != v24 );
LABEL_34:
      v24 = (unsigned __int64)v54;
LABEL_35:
      if ( (_BYTE *)v24 != v56 )
        _libc_free(v24);
      return;
    }
    v27 = v9;
    v9 = v10;
    v10 = v27;
  }
  if ( v8 == 42 )
  {
    sub_2DEBFC0(v10, a2);
    if ( *((_DWORD *)v9 + 8) == *(_DWORD *)(a2 + 136) )
    {
      sub_C45EE0(a2 + 128, (__int64 *)v9 + 3);
      return;
    }
LABEL_58:
    *(_DWORD *)a2 = -1;
    return;
  }
  if ( v8 != 55 )
    goto LABEL_46;
  sub_2DEBFC0(v10, a2);
  v13 = *((_DWORD *)v9 + 8);
  v14 = *(_DWORD *)(a2 + 136);
  if ( v13 != v14 )
    goto LABEL_58;
  v15 = (const void **)(v9 + 24);
  if ( v13 <= 0x40 )
  {
    if ( !*((_QWORD *)v9 + 3) )
      return;
  }
  else
  {
    v15 = (const void **)(v9 + 24);
    if ( v13 == (unsigned int)sub_C444A0((__int64)(v9 + 24)) )
      return;
  }
  v16 = (_QWORD *)*((_QWORD *)v9 + 3);
  if ( v13 <= 0x40 )
  {
    v17 = *((_QWORD *)v9 + 3);
    if ( (unsigned int)v16 < v13 )
    {
LABEL_11:
      if ( v14 <= 0x40 )
      {
        _RAX = *(_QWORD *)(a2 + 128);
        __asm { tzcnt   rdx, rax }
        v47 = _RAX == 0;
        v18 = 64;
        if ( !v47 )
          v18 = _RDX;
        if ( v14 <= v18 )
          v18 = v14;
      }
      else
      {
        v18 = sub_C44590(a2 + 128);
      }
      if ( v18 >= v17 )
      {
        if ( *(_DWORD *)a2 != -1 )
        {
          v35 = v17 + *(_DWORD *)a2;
          if ( v14 < v35 )
            v35 = v14;
          *(_DWORD *)a2 = v35;
        }
      }
      else
      {
        *(_DWORD *)a2 = v14;
      }
      if ( *(_QWORD *)(a2 + 8) )
      {
        LODWORD(v52) = 0;
        LODWORD(v54) = *((_DWORD *)v9 + 8);
        if ( (unsigned int)v54 > 0x40 )
          sub_C43780((__int64)&v53, v15);
        else
          v53 = *((_QWORD *)v9 + 3);
        v36 = *(unsigned int *)(a2 + 24);
        v37 = *(unsigned int *)(a2 + 28);
        v38 = v36 + 1;
        v39 = v36;
        if ( v36 + 1 > v37 )
        {
          v48 = *(_QWORD *)(a2 + 16);
          v41 = &v52;
          v49 = a2 + 16;
          if ( v48 > (unsigned __int64)&v52 || (unsigned __int64)&v52 >= v48 + 24 * v36 )
          {
            sub_2DEABD0(v49, v38, v36, v37, v11, v12);
            v36 = *(unsigned int *)(a2 + 24);
            v40 = *(_QWORD *)(a2 + 16);
            v39 = *(_DWORD *)(a2 + 24);
          }
          else
          {
            sub_2DEABD0(v49, v38, v36, v37, v11, v12);
            v40 = *(_QWORD *)(a2 + 16);
            v36 = *(unsigned int *)(a2 + 24);
            v41 = (unsigned __int64 *)((char *)&v52 + v40 - v48);
            v39 = *(_DWORD *)(a2 + 24);
          }
        }
        else
        {
          v40 = *(_QWORD *)(a2 + 16);
          v41 = &v52;
        }
        v42 = v40 + 24 * v36;
        if ( v42 )
        {
          *(_DWORD *)v42 = *(_DWORD *)v41;
          v43 = *((_DWORD *)v41 + 4);
          *((_DWORD *)v41 + 4) = 0;
          *(_DWORD *)(v42 + 16) = v43;
          *(_QWORD *)(v42 + 8) = v41[1];
          v39 = *(_DWORD *)(a2 + 24);
        }
        v44 = (unsigned int)v54 <= 0x40;
        *(_DWORD *)(a2 + 24) = v39 + 1;
        if ( !v44 && v53 )
          j_j___libc_free_0_0(v53);
        v14 = *(_DWORD *)(a2 + 136);
      }
      LODWORD(v53) = v14;
      if ( v14 > 0x40 )
      {
        sub_C43780((__int64)&v52, (const void **)(a2 + 128));
        v14 = v53;
        if ( (unsigned int)v53 > 0x40 )
        {
          sub_C482E0((__int64)&v52, v17);
LABEL_20:
          if ( *(_DWORD *)(a2 + 136) > 0x40u )
          {
            v19 = *(_QWORD *)(a2 + 128);
            if ( v19 )
              j_j___libc_free_0_0(v19);
          }
          *(_QWORD *)(a2 + 128) = v52;
          *(_DWORD *)(a2 + 136) = v53;
          return;
        }
      }
      else
      {
        v52 = *(_QWORD *)(a2 + 128);
      }
      if ( v14 == v17 )
        v52 = 0;
      else
        v52 >>= v17;
      goto LABEL_20;
    }
    LODWORD(v53) = v13;
    v52 = 0;
  }
  else
  {
    v17 = *v16;
    if ( v13 > v17 )
      goto LABEL_11;
    LODWORD(v53) = v13;
    sub_C43690((__int64)&v52, 0, 0);
  }
  sub_2DEB8E0(a2, (__int64)&v52);
  if ( (unsigned int)v53 > 0x40 && v52 )
    j_j___libc_free_0_0(v52);
}
