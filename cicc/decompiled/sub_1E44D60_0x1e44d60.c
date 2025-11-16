// Function: sub_1E44D60
// Address: 0x1e44d60
//
void __fastcall sub_1E44D60(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // r10
  signed __int64 v6; // r9
  char *v7; // r8
  char *v8; // rbx
  __int64 v9; // r13
  char *v10; // r12
  char *v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // r10
  __int64 v16; // r14
  __int64 v17; // rax
  char *v18; // rax
  unsigned int v19; // ecx
  unsigned int v20; // ebx
  bool v21; // dl
  unsigned int v22; // edx
  int v23; // esi
  int v24; // r12d
  bool v25; // cl
  unsigned int v26; // ecx
  int v27; // edi
  __int64 v28; // rsi
  __int64 v29; // r14
  __int64 v30; // r13
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rcx
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+10h] [rbp-70h]
  int v42; // [rsp+1Ch] [rbp-64h]
  __int64 v43; // [rsp+20h] [rbp-60h]
  __int64 v44; // [rsp+28h] [rbp-58h]
  __int64 v46; // [rsp+30h] [rbp-50h]
  __int64 v47; // [rsp+38h] [rbp-48h]
  __int64 v48; // [rsp+38h] [rbp-48h]
  __int64 v49; // [rsp+38h] [rbp-48h]
  int v50; // [rsp+38h] [rbp-48h]
  __int64 v51; // [rsp+40h] [rbp-40h]
  __int64 v52; // [rsp+40h] [rbp-40h]
  __int64 v53; // [rsp+40h] [rbp-40h]
  __int64 v54; // [rsp+40h] [rbp-40h]
  char *v55; // [rsp+48h] [rbp-38h]
  __int64 v56; // [rsp+48h] [rbp-38h]
  char *v57; // [rsp+48h] [rbp-38h]
  char v58; // [rsp+48h] [rbp-38h]

  if ( !a4 || !a5 )
    return;
  if ( a4 + a5 == 2 )
  {
    v17 = (__int64)a1;
    v14 = (__int64)a2;
LABEL_12:
    v19 = *(_DWORD *)(v14 + 60);
    v20 = *(_DWORD *)(v17 + 60);
    v21 = v19 > v20;
    if ( v19 == v20
      && ((v22 = *(_DWORD *)(v14 + 72)) == 0 || (v26 = *(_DWORD *)(v17 + 72), v22 == v26) || (v21 = v22 < v26, !v26)) )
    {
      v23 = *(_DWORD *)(v14 + 64);
      v24 = *(_DWORD *)(v17 + 64);
      v25 = v23 < v24;
      if ( v23 == v24 )
        v25 = *(_DWORD *)(v14 + 68) > *(_DWORD *)(v17 + 68);
      if ( !v25 )
        return;
    }
    else
    {
      if ( !v21 )
        return;
      v24 = *(_DWORD *)(v17 + 64);
    }
    v27 = *(_DWORD *)(v17 + 24);
    v28 = *(_QWORD *)(v17 + 48);
    *(_DWORD *)(v17 + 24) = 0;
    v29 = *(_QWORD *)(v17 + 8);
    v30 = *(_QWORD *)(v17 + 16);
    *(_QWORD *)(v17 + 8) = 0;
    v42 = v27;
    v31 = *(_QWORD *)(v17 + 32);
    ++*(_QWORD *)v17;
    v46 = v31;
    v43 = v28;
    v32 = *(_QWORD *)(v17 + 68);
    v44 = *(_QWORD *)(v17 + 40);
    LOBYTE(v31) = *(_BYTE *)(v17 + 56);
    *(_QWORD *)(v17 + 16) = 0;
    *(_QWORD *)(v17 + 48) = 0;
    *(_QWORD *)(v17 + 40) = 0;
    *(_QWORD *)(v17 + 32) = 0;
    v58 = v31;
    v41 = v32;
    v54 = *(_QWORD *)(v17 + 80);
    v40 = v17;
    v50 = *(_DWORD *)(v17 + 88);
    j___libc_free_0(0);
    v33 = v40;
    *(_QWORD *)(v40 + 16) = 0;
    *(_QWORD *)(v40 + 8) = 0;
    *(_DWORD *)(v40 + 24) = 0;
    ++*(_QWORD *)v40;
    v34 = *(_QWORD *)(v14 + 8);
    ++*(_QWORD *)v14;
    v35 = *(_QWORD *)(v40 + 8);
    *(_QWORD *)(v40 + 8) = v34;
    LODWORD(v34) = *(_DWORD *)(v14 + 16);
    *(_QWORD *)(v14 + 8) = v35;
    LODWORD(v35) = *(_DWORD *)(v40 + 16);
    *(_DWORD *)(v40 + 16) = v34;
    LODWORD(v34) = *(_DWORD *)(v14 + 20);
    *(_DWORD *)(v14 + 16) = v35;
    LODWORD(v35) = *(_DWORD *)(v40 + 20);
    *(_DWORD *)(v40 + 20) = v34;
    LODWORD(v34) = *(_DWORD *)(v14 + 24);
    *(_DWORD *)(v14 + 20) = v35;
    LODWORD(v35) = *(_DWORD *)(v40 + 24);
    *(_DWORD *)(v40 + 24) = v34;
    *(_DWORD *)(v14 + 24) = v35;
    v36 = *(_QWORD *)(v40 + 32);
    v37 = *(_QWORD *)(v40 + 48);
    *(_QWORD *)(v40 + 32) = *(_QWORD *)(v14 + 32);
    *(_QWORD *)(v40 + 40) = *(_QWORD *)(v14 + 40);
    *(_QWORD *)(v40 + 48) = *(_QWORD *)(v14 + 48);
    *(_QWORD *)(v14 + 32) = 0;
    *(_QWORD *)(v14 + 40) = 0;
    *(_QWORD *)(v14 + 48) = 0;
    if ( v36 )
    {
      j_j___libc_free_0(v36, v37 - v36);
      v33 = v40;
    }
    *(_BYTE *)(v33 + 56) = *(_BYTE *)(v14 + 56);
    *(_DWORD *)(v33 + 60) = *(_DWORD *)(v14 + 60);
    *(_DWORD *)(v33 + 64) = *(_DWORD *)(v14 + 64);
    *(_DWORD *)(v33 + 68) = *(_DWORD *)(v14 + 68);
    *(_DWORD *)(v33 + 72) = *(_DWORD *)(v14 + 72);
    *(_QWORD *)(v33 + 80) = *(_QWORD *)(v14 + 80);
    *(_DWORD *)(v33 + 88) = *(_DWORD *)(v14 + 88);
    j___libc_free_0(*(_QWORD *)(v14 + 8));
    v38 = *(_QWORD *)(v14 + 32);
    *(_QWORD *)(v14 + 8) = v29;
    v39 = *(_QWORD *)(v14 + 48);
    ++*(_QWORD *)v14;
    *(_DWORD *)(v14 + 24) = v42;
    *(_QWORD *)(v14 + 16) = v30;
    *(_QWORD *)(v14 + 32) = v46;
    *(_QWORD *)(v14 + 40) = v44;
    *(_QWORD *)(v14 + 48) = v43;
    if ( v38 )
      j_j___libc_free_0(v38, v39 - v38);
    *(_DWORD *)(v14 + 60) = v20;
    *(_DWORD *)(v14 + 64) = v24;
    *(_BYTE *)(v14 + 56) = v58;
    *(_QWORD *)(v14 + 68) = v41;
    *(_QWORD *)(v14 + 80) = v54;
    *(_DWORD *)(v14 + 88) = v50;
    j___libc_free_0(0);
    return;
  }
  v5 = a5;
  v6 = a4;
  v7 = a2;
  v8 = a1;
  if ( v5 >= a4 )
    goto LABEL_10;
LABEL_5:
  v47 = v5;
  v51 = v6;
  v55 = v7;
  v9 = v6 / 2;
  v10 = &v8[32 * (v6 / 2) + 32 * ((v6 + ((unsigned __int64)v6 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
  v11 = (char *)sub_1E426C0(v7, a3, v10);
  v12 = (__int64)v55;
  v13 = v51;
  v14 = (__int64)v11;
  v15 = v47;
  v16 = 0xAAAAAAAAAAAAAAABLL * ((v11 - v55) >> 5);
  while ( 1 )
  {
    v52 = v15;
    v56 = v13;
    v48 = sub_1E435D0((__int64)v10, v12, v14);
    sub_1E44D60(v8, v10, v48, v9, v16);
    v5 = v52 - v16;
    v6 = v56 - v9;
    if ( v56 == v9 )
      break;
    v17 = v48;
    if ( !v5 )
      break;
    if ( v5 + v6 == 2 )
      goto LABEL_12;
    v8 = (char *)v48;
    v7 = (char *)v14;
    if ( v5 < v6 )
      goto LABEL_5;
LABEL_10:
    v49 = v6;
    v53 = v5;
    v57 = v7;
    v16 = v5 / 2;
    v14 = (__int64)&v7[32 * (v5 / 2) + 32 * ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
    v18 = (char *)sub_1E42610(v8, (__int64)v7, (_DWORD *)v14);
    v13 = v49;
    v15 = v53;
    v10 = v18;
    v12 = (__int64)v57;
    v9 = 0xAAAAAAAAAAAAAAABLL * ((v18 - v8) >> 5);
  }
}
