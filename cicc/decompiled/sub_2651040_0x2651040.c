// Function: sub_2651040
// Address: 0x2651040
//
void __fastcall sub_2651040(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  bool v5; // r8
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rcx
  unsigned __int64 v13; // rbx
  __int64 v14; // r13
  unsigned __int64 i; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rsi
  _QWORD *v26; // rdi
  _QWORD *v27; // r14
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  char *v34; // rcx
  _BYTE *v35; // rsi
  signed __int64 v36; // r8
  __int64 v37; // rax
  unsigned __int64 v38; // r8
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rax
  char *v42; // rdx
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // rdi
  __int64 v47; // rsi
  __int64 v48; // rdi
  __int64 v49; // rax
  int v50; // [rsp+14h] [rbp-DCh]
  int v51; // [rsp+18h] [rbp-D8h]
  int v52; // [rsp+1Ch] [rbp-D4h]
  __int64 v53; // [rsp+20h] [rbp-D0h]
  __int64 v54; // [rsp+28h] [rbp-C8h]
  __int64 v55; // [rsp+30h] [rbp-C0h]
  __int64 v56; // [rsp+38h] [rbp-B8h]
  __int64 v57; // [rsp+40h] [rbp-B0h]
  __int64 v59; // [rsp+58h] [rbp-98h] BYREF
  __int64 v60; // [rsp+68h] [rbp-88h] BYREF
  __int64 v61; // [rsp+70h] [rbp-80h] BYREF
  _QWORD *v62; // [rsp+78h] [rbp-78h]
  _QWORD *v63; // [rsp+80h] [rbp-70h]
  __int64 v64; // [rsp+88h] [rbp-68h]
  __int64 v65; // [rsp+90h] [rbp-60h]
  __int64 v66; // [rsp+98h] [rbp-58h]
  __int64 v67; // [rsp+A0h] [rbp-50h]
  __int64 v68; // [rsp+A8h] [rbp-48h]
  int v69; // [rsp+B0h] [rbp-40h]

  v59 = a3;
  if ( a1 == a2 )
    return;
  v3 = a1 + 72;
  if ( a1 + 72 == a2 )
    return;
  do
  {
    v5 = sub_2650F70(&v59, v3, a1);
    v6 = v3;
    v3 += 72;
    if ( v5 )
    {
      v7 = *(_QWORD *)(v3 - 64);
      v8 = v6 - a1;
      v9 = *(_QWORD *)(v3 - 72);
      *(_QWORD *)(v3 - 64) = 0;
      ++*(_QWORD *)(v3 - 32);
      v57 = v7;
      v10 = *(_QWORD *)(v3 - 56);
      *(_QWORD *)(v3 - 56) = 0;
      v56 = v10;
      v11 = *(_QWORD *)(v3 - 48);
      *(_QWORD *)(v3 - 48) = 0;
      v55 = v11;
      v54 = *(_QWORD *)(v3 - 40);
      v12 = *(_QWORD *)(v3 - 24);
      *(_QWORD *)(v3 - 24) = 0;
      v53 = v12;
      LODWORD(v12) = *(_DWORD *)(v3 - 16);
      *(_DWORD *)(v3 - 16) = 0;
      v52 = v12;
      LODWORD(v12) = *(_DWORD *)(v3 - 12);
      *(_DWORD *)(v3 - 12) = 0;
      v51 = v12;
      LODWORD(v12) = *(_DWORD *)(v3 - 8);
      *(_DWORD *)(v3 - 8) = 0;
      v50 = v12;
      v13 = 0x8E38E38E38E38E39LL * (v8 >> 3);
      if ( v8 > 0 )
      {
        v14 = v3;
        for ( i = 0; ; i = *(_QWORD *)(v14 - 64) )
        {
          v16 = *(_QWORD *)(v14 - 144);
          v14 -= 72;
          *(_QWORD *)v14 = v16;
          v17 = *(_QWORD *)(v14 - 64);
          *(_QWORD *)(v14 - 64) = 0;
          *(_QWORD *)(v14 + 8) = v17;
          v18 = *(_QWORD *)(v14 - 56);
          *(_QWORD *)(v14 - 56) = 0;
          *(_QWORD *)(v14 + 16) = v18;
          v19 = *(_QWORD *)(v14 - 48);
          *(_QWORD *)(v14 - 48) = 0;
          *(_QWORD *)(v14 + 24) = v19;
          if ( i )
            j_j___libc_free_0(i);
          v20 = *(unsigned int *)(v14 + 64);
          v21 = *(_QWORD *)(v14 + 48);
          *(_QWORD *)(v14 + 32) = *(_QWORD *)(v14 - 40);
          sub_C7D6A0(v21, 4 * v20, 4);
          v22 = *(_QWORD *)(v14 - 24);
          ++*(_QWORD *)(v14 + 40);
          ++*(_QWORD *)(v14 - 32);
          *(_QWORD *)(v14 + 48) = v22;
          LODWORD(v22) = *(_DWORD *)(v14 - 16);
          *(_QWORD *)(v14 - 24) = 0;
          *(_DWORD *)(v14 + 56) = v22;
          LODWORD(v22) = *(_DWORD *)(v14 - 12);
          *(_DWORD *)(v14 - 16) = 0;
          *(_DWORD *)(v14 + 60) = v22;
          LODWORD(v22) = *(_DWORD *)(v14 - 8);
          *(_DWORD *)(v14 - 12) = 0;
          *(_DWORD *)(v14 + 64) = v22;
          *(_DWORD *)(v14 - 8) = 0;
          if ( !--v13 )
            break;
        }
      }
      v23 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)a1 = v9;
      *(_QWORD *)(a1 + 8) = v57;
      *(_QWORD *)(a1 + 16) = v56;
      *(_QWORD *)(a1 + 24) = v55;
      if ( v23 )
        j_j___libc_free_0(v23);
      v24 = *(_QWORD *)(a1 + 48);
      v25 = 4LL * *(unsigned int *)(a1 + 64);
      *(_QWORD *)(a1 + 32) = v54;
      sub_C7D6A0(v24, v25, 4);
      ++*(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 48) = v53;
      *(_DWORD *)(a1 + 56) = v52;
      *(_DWORD *)(a1 + 60) = v51;
      *(_DWORD *)(a1 + 64) = v50;
      sub_C7D6A0(0, 0, 4);
      continue;
    }
    v26 = *(_QWORD **)(v3 - 64);
    v27 = *(_QWORD **)(v3 - 56);
    *(_QWORD *)(v3 - 64) = 0;
    v28 = v3 - 144;
    v29 = v59;
    ++*(_QWORD *)(v3 - 32);
    v62 = v26;
    v60 = v29;
    v30 = *(_QWORD *)(v3 - 72);
    v63 = v27;
    v61 = v30;
    v31 = *(_QWORD *)(v3 - 48);
    *(_QWORD *)(v3 - 56) = 0;
    v64 = v31;
    v32 = *(_QWORD *)(v3 - 40);
    *(_QWORD *)(v3 - 48) = 0;
    v65 = v32;
    v33 = *(_QWORD *)(v3 - 24);
    v66 = 1;
    v67 = v33;
    LODWORD(v33) = *(_DWORD *)(v3 - 16);
    *(_QWORD *)(v3 - 24) = 0;
    LODWORD(v68) = v33;
    LODWORD(v33) = *(_DWORD *)(v3 - 12);
    *(_DWORD *)(v3 - 16) = 0;
    HIDWORD(v68) = v33;
    LODWORD(v33) = *(_DWORD *)(v3 - 8);
    *(_DWORD *)(v3 - 12) = 0;
    v69 = v33;
    *(_DWORD *)(v3 - 8) = 0;
    while ( 1 )
    {
      v34 = *(char **)(v28 + 16);
      v35 = *(_BYTE **)(v28 + 8);
      v36 = (char *)v27 - (char *)v26;
      if ( v34 - v35 < (unsigned __int64)((char *)v27 - (char *)v26) )
        goto LABEL_27;
      if ( v36 != v34 - v35 )
        goto LABEL_18;
      v42 = *(char **)(v28 + 8);
      if ( v26 == v27 )
        break;
      v43 = v26;
      while ( *v43 >= *(_QWORD *)v42 )
      {
        if ( *v43 > *(_QWORD *)v42 )
          goto LABEL_31;
        ++v43;
        v42 += 8;
        if ( v27 == v43 )
          goto LABEL_30;
      }
LABEL_27:
      v44 = *(_QWORD *)v28;
      v45 = *(_QWORD *)(v28 + 24);
      *(_QWORD *)(v28 + 88) = v34;
      v46 = *(_QWORD *)(v28 + 80);
      *(_QWORD *)(v28 + 8) = 0;
      *(_QWORD *)(v28 + 72) = v44;
      *(_QWORD *)(v28 + 80) = v35;
      *(_QWORD *)(v28 + 96) = v45;
      *(_QWORD *)(v28 + 16) = 0;
      *(_QWORD *)(v28 + 24) = 0;
      if ( v46 )
        j_j___libc_free_0(v46);
      v47 = *(unsigned int *)(v28 + 136);
      v48 = *(_QWORD *)(v28 + 120);
      *(_QWORD *)(v28 + 104) = *(_QWORD *)(v28 + 32);
      sub_C7D6A0(v48, 4 * v47, 4);
      v49 = *(_QWORD *)(v28 + 48);
      ++*(_QWORD *)(v28 + 112);
      v27 = v63;
      v26 = v62;
      *(_QWORD *)(v28 + 48) = 0;
      *(_QWORD *)(v28 + 120) = v49;
      LODWORD(v49) = *(_DWORD *)(v28 + 56);
      ++*(_QWORD *)(v28 + 40);
      v28 -= 72;
      *(_DWORD *)(v28 + 200) = v49;
      LODWORD(v49) = *(_DWORD *)(v28 + 132);
      *(_DWORD *)(v28 + 128) = 0;
      *(_DWORD *)(v28 + 204) = v49;
      LODWORD(v49) = *(_DWORD *)(v28 + 136);
      *(_DWORD *)(v28 + 132) = 0;
      *(_DWORD *)(v28 + 208) = v49;
      *(_DWORD *)(v28 + 136) = 0;
    }
LABEL_30:
    if ( v34 != v42 )
      goto LABEL_27;
LABEL_31:
    if ( v36 && memcmp(v26, v35, (char *)v27 - (char *)v26) )
      goto LABEL_18;
    if ( sub_2650CE0(&v60, (__int64)&v61, v28) )
    {
      v34 = *(char **)(v28 + 16);
      v35 = *(_BYTE **)(v28 + 8);
      goto LABEL_27;
    }
    v26 = v62;
    v27 = v63;
LABEL_18:
    v37 = v61;
    v38 = *(_QWORD *)(v28 + 80);
    *(_QWORD *)(v28 + 88) = v27;
    *(_QWORD *)(v28 + 80) = v26;
    *(_QWORD *)(v28 + 72) = v37;
    v62 = 0;
    *(_QWORD *)(v28 + 96) = v64;
    v63 = 0;
    v64 = 0;
    if ( v38 )
      j_j___libc_free_0(v38);
    v39 = *(unsigned int *)(v28 + 136);
    v40 = *(_QWORD *)(v28 + 120);
    *(_QWORD *)(v28 + 104) = v65;
    sub_C7D6A0(v40, 4 * v39, 4);
    v41 = v67;
    ++*(_QWORD *)(v28 + 112);
    ++v66;
    *(_QWORD *)(v28 + 120) = v41;
    v67 = 0;
    *(_QWORD *)(v28 + 128) = v68;
    v68 = 0;
    *(_DWORD *)(v28 + 136) = v69;
    v69 = 0;
    sub_C7D6A0(0, 0, 4);
    if ( v62 )
      j_j___libc_free_0((unsigned __int64)v62);
  }
  while ( a2 != v3 );
}
