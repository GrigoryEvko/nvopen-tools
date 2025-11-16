// Function: sub_1E44500
// Address: 0x1e44500
//
void __fastcall sub_1E44500(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // r12d
  unsigned int v5; // edx
  bool v6; // si
  __int64 v7; // r15
  __int64 v8; // rcx
  unsigned __int64 v9; // rbx
  __int64 i; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // r8
  __int64 v22; // rdi
  __int64 v23; // rsi
  int v24; // eax
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  char v29; // al
  unsigned int v30; // eax
  bool v31; // dl
  unsigned int v32; // eax
  int v33; // edx
  bool v34; // al
  __int64 v35; // rdi
  __int64 v36; // rsi
  unsigned int v37; // edx
  int v38; // edx
  unsigned int v40; // [rsp+8h] [rbp-78h]
  int v41; // [rsp+Ch] [rbp-74h]
  unsigned int v42; // [rsp+10h] [rbp-70h]
  int v43; // [rsp+14h] [rbp-6Ch]
  __int64 v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+20h] [rbp-60h]
  __int64 v46; // [rsp+28h] [rbp-58h]
  __int64 v47; // [rsp+30h] [rbp-50h]
  int v48; // [rsp+38h] [rbp-48h]
  int v49; // [rsp+3Ch] [rbp-44h]
  __int64 v50; // [rsp+40h] [rbp-40h]
  char v51; // [rsp+4Bh] [rbp-35h]
  int v52; // [rsp+4Ch] [rbp-34h]

  if ( a1 == a2 )
    return;
  v2 = a1 + 96;
  if ( a2 == a1 + 96 )
    return;
  while ( 1 )
  {
    v4 = *(_DWORD *)(v2 + 60);
    v5 = *(_DWORD *)(a1 + 60);
    v42 = *(_DWORD *)(v2 + 72);
    v41 = *(_DWORD *)(v2 + 64);
    v6 = v4 > v5;
    v40 = *(_DWORD *)(v2 + 68);
    if ( v4 == v5 )
    {
      if ( !v42 || (v37 = *(_DWORD *)(a1 + 72)) == 0 || (v6 = v37 > v42, v37 == v42) )
      {
        v38 = *(_DWORD *)(a1 + 64);
        v6 = v38 > v41;
        if ( v38 == v41 )
          v6 = *(_DWORD *)(a1 + 68) < v40;
      }
    }
    v7 = v2 + 96;
    v51 = *(_BYTE *)(v2 + 56);
    v50 = *(_QWORD *)(v2 + 80);
    v49 = *(_DWORD *)(v2 + 88);
    v47 = *(_QWORD *)(v2 + 32);
    v46 = *(_QWORD *)(v2 + 40);
    v45 = *(_QWORD *)(v2 + 48);
    v8 = *(_QWORD *)v2 + 1LL;
    v44 = *(_QWORD *)(v2 + 8);
    v48 = *(_DWORD *)(v2 + 16);
    v43 = *(_DWORD *)(v2 + 20);
    v52 = *(_DWORD *)(v2 + 24);
    if ( !v6 )
      break;
    *(_QWORD *)v2 = v8;
    *(_QWORD *)(v2 + 8) = 0;
    *(_DWORD *)(v2 + 16) = 0;
    *(_DWORD *)(v2 + 20) = 0;
    *(_DWORD *)(v2 + 24) = 0;
    v9 = 0xAAAAAAAAAAAAAAABLL * ((v2 - a1) >> 5);
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 32) = 0;
    if ( v2 - a1 > 0 )
    {
      for ( i = 0; ; i = *(_QWORD *)(v2 + 8) )
      {
        j___libc_free_0(i);
        v11 = *(_QWORD *)(v2 - 88);
        v12 = *(_QWORD *)(v2 + 32);
        v2 -= 96;
        v13 = *(_QWORD *)(v2 + 144);
        ++*(_QWORD *)(v2 + 96);
        *(_QWORD *)(v2 + 104) = v11;
        LODWORD(v11) = *(_DWORD *)(v2 + 16);
        ++*(_QWORD *)v2;
        *(_DWORD *)(v2 + 112) = v11;
        LODWORD(v11) = *(_DWORD *)(v2 + 20);
        *(_DWORD *)(v2 + 16) = 0;
        *(_DWORD *)(v2 + 116) = v11;
        LODWORD(v11) = *(_DWORD *)(v2 + 24);
        *(_DWORD *)(v2 + 20) = 0;
        *(_DWORD *)(v2 + 120) = v11;
        v14 = *(_QWORD *)(v2 + 32);
        *(_DWORD *)(v2 + 24) = 0;
        *(_QWORD *)(v2 + 128) = v14;
        v15 = *(_QWORD *)(v2 + 40);
        *(_QWORD *)(v2 + 32) = 0;
        *(_QWORD *)(v2 + 136) = v15;
        v16 = *(_QWORD *)(v2 + 48);
        *(_QWORD *)(v2 + 40) = 0;
        *(_QWORD *)(v2 + 144) = v16;
        *(_QWORD *)(v2 + 48) = 0;
        *(_QWORD *)(v2 + 8) = 0;
        if ( v12 )
          j_j___libc_free_0(v12, v13 - v12);
        *(_BYTE *)(v2 + 152) = *(_BYTE *)(v2 + 56);
        *(_DWORD *)(v2 + 156) = *(_DWORD *)(v2 + 60);
        *(_DWORD *)(v2 + 160) = *(_DWORD *)(v2 + 64);
        *(_DWORD *)(v2 + 164) = *(_DWORD *)(v2 + 68);
        *(_DWORD *)(v2 + 168) = *(_DWORD *)(v2 + 72);
        *(_QWORD *)(v2 + 176) = *(_QWORD *)(v2 + 80);
        *(_DWORD *)(v2 + 184) = *(_DWORD *)(v2 + 88);
        if ( !--v9 )
          break;
      }
    }
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    v17 = *(_QWORD *)(a1 + 32);
    v18 = *(_QWORD *)(a1 + 48);
    ++*(_QWORD *)a1;
    *(_QWORD *)(a1 + 8) = v44;
    *(_DWORD *)(a1 + 16) = v48;
    *(_DWORD *)(a1 + 20) = v43;
    *(_DWORD *)(a1 + 24) = v52;
    *(_QWORD *)(a1 + 32) = v47;
    *(_QWORD *)(a1 + 40) = v46;
    *(_QWORD *)(a1 + 48) = v45;
    if ( v17 )
      j_j___libc_free_0(v17, v18 - v17);
    *(_DWORD *)(a1 + 60) = v4;
    *(_BYTE *)(a1 + 56) = v51;
    *(_DWORD *)(a1 + 64) = v41;
    *(_DWORD *)(a1 + 68) = v40;
    *(_DWORD *)(a1 + 72) = v42;
    *(_QWORD *)(a1 + 80) = v50;
    *(_DWORD *)(a1 + 88) = v49;
    j___libc_free_0(0);
    if ( a2 == v7 )
      return;
LABEL_14:
    v2 = v7;
  }
  *(_QWORD *)v2 = v8;
  v19 = 0;
  *(_QWORD *)(v2 + 8) = 0;
  *(_DWORD *)(v2 + 16) = 0;
  *(_DWORD *)(v2 + 20) = 0;
  *(_DWORD *)(v2 + 24) = 0;
  *(_QWORD *)(v2 + 48) = 0;
  *(_QWORD *)(v2 + 40) = 0;
  *(_QWORD *)(v2 + 32) = 0;
  while ( 1 )
  {
    v30 = *(_DWORD *)(v2 - 36);
    v31 = v4 > v30;
    if ( v4 == v30 )
    {
      if ( !v42 )
        break;
      v32 = *(_DWORD *)(v2 - 24);
      if ( v32 == v42 )
        break;
      v31 = v32 > v42;
      if ( !v32 )
        break;
    }
    if ( !v31 )
      goto LABEL_27;
LABEL_17:
    j___libc_free_0(v19);
    v20 = *(_DWORD *)(v2 - 80);
    v21 = *(_QWORD *)(v2 + 32);
    *(_DWORD *)(v2 - 80) = 0;
    v22 = *(_QWORD *)(v2 - 88);
    v23 = *(_QWORD *)(v2 + 48);
    *(_QWORD *)(v2 - 88) = 0;
    *(_DWORD *)(v2 + 16) = v20;
    v24 = *(_DWORD *)(v2 - 76);
    *(_QWORD *)(v2 + 8) = v22;
    v19 = 0;
    *(_DWORD *)(v2 + 20) = v24;
    v25 = *(_DWORD *)(v2 - 72);
    ++*(_QWORD *)v2;
    *(_DWORD *)(v2 + 24) = v25;
    v26 = *(_QWORD *)(v2 - 64);
    ++*(_QWORD *)(v2 - 96);
    *(_QWORD *)(v2 + 32) = v26;
    v27 = *(_QWORD *)(v2 - 56);
    *(_DWORD *)(v2 - 76) = 0;
    *(_QWORD *)(v2 + 40) = v27;
    v28 = *(_QWORD *)(v2 - 48);
    *(_DWORD *)(v2 - 72) = 0;
    *(_QWORD *)(v2 + 48) = v28;
    *(_QWORD *)(v2 - 64) = 0;
    *(_QWORD *)(v2 - 56) = 0;
    *(_QWORD *)(v2 - 48) = 0;
    if ( v21 )
    {
      j_j___libc_free_0(v21, v23 - v21);
      v19 = *(_QWORD *)(v2 - 88);
    }
    v29 = *(_BYTE *)(v2 - 40);
    v2 -= 96;
    *(_BYTE *)(v2 + 152) = v29;
    *(_DWORD *)(v2 + 156) = *(_DWORD *)(v2 + 60);
    *(_DWORD *)(v2 + 160) = *(_DWORD *)(v2 + 64);
    *(_DWORD *)(v2 + 164) = *(_DWORD *)(v2 + 68);
    *(_DWORD *)(v2 + 168) = *(_DWORD *)(v2 + 72);
    *(_QWORD *)(v2 + 176) = *(_QWORD *)(v2 + 80);
    *(_DWORD *)(v2 + 184) = *(_DWORD *)(v2 + 88);
  }
  v33 = *(_DWORD *)(v2 - 32);
  v34 = v33 > v41;
  if ( v33 == v41 )
    v34 = *(_DWORD *)(v2 - 28) < v40;
  if ( v34 )
    goto LABEL_17;
LABEL_27:
  j___libc_free_0(v19);
  v35 = *(_QWORD *)(v2 + 32);
  v36 = *(_QWORD *)(v2 + 48);
  ++*(_QWORD *)v2;
  *(_QWORD *)(v2 + 8) = v44;
  *(_DWORD *)(v2 + 16) = v48;
  *(_DWORD *)(v2 + 20) = v43;
  *(_DWORD *)(v2 + 24) = v52;
  *(_QWORD *)(v2 + 32) = v47;
  *(_QWORD *)(v2 + 40) = v46;
  *(_QWORD *)(v2 + 48) = v45;
  if ( v35 )
    j_j___libc_free_0(v35, v36 - v35);
  *(_DWORD *)(v2 + 60) = v4;
  *(_BYTE *)(v2 + 56) = v51;
  *(_DWORD *)(v2 + 64) = v41;
  *(_DWORD *)(v2 + 68) = v40;
  *(_DWORD *)(v2 + 72) = v42;
  *(_QWORD *)(v2 + 80) = v50;
  *(_DWORD *)(v2 + 88) = v49;
  j___libc_free_0(0);
  if ( a2 != v7 )
    goto LABEL_14;
}
