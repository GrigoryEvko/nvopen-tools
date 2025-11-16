// Function: sub_1BAA750
// Address: 0x1baa750
//
__int64 __fastcall sub_1BAA750(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // r13
  char *v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // r9
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r15
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r15
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // r14
  _BYTE *v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rdi
  __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-88h]
  __int64 v38; // [rsp+10h] [rbp-80h]
  _QWORD *v39; // [rsp+10h] [rbp-80h]
  _QWORD *v40; // [rsp+10h] [rbp-80h]
  _QWORD *v41; // [rsp+10h] [rbp-80h]
  _QWORD *v42; // [rsp+10h] [rbp-80h]
  _QWORD *v43; // [rsp+10h] [rbp-80h]
  __int64 *v45; // [rsp+20h] [rbp-70h] BYREF
  char *v46; // [rsp+28h] [rbp-68h]
  __int16 v47; // [rsp+30h] [rbp-60h]
  __int64 v48[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v49; // [rsp+50h] [rbp-40h] BYREF

  v4 = sub_1BA9430(a1, *(_QWORD *)(a2 + 40), a4);
  v5 = sub_15F29F0((unsigned int)*(unsigned __int8 *)(a2 + 16) - 24);
  if ( *v5 )
  {
    v46 = v5;
    v45 = (__int64 *)"pred.";
    v47 = 771;
  }
  else
  {
    v45 = (__int64 *)"pred.";
    v47 = 259;
  }
  sub_16E2FC0(v48, (__int64)&v45);
  v6 = sub_22077B0(48);
  v7 = (_QWORD *)v6;
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = 0;
    *(_QWORD *)(v6 + 16) = 0;
    *(_BYTE *)(v6 + 24) = 1;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 40) = 0;
    *(_QWORD *)v6 = &unk_49F6FD8;
    if ( v4 )
    {
      v38 = v6;
      v8 = sub_22077B0(72);
      v7 = (_QWORD *)v38;
      v9 = v8;
      if ( v8 )
      {
        *(_BYTE *)v8 = 1;
        *(_QWORD *)(v8 + 8) = v8 + 24;
        *(_QWORD *)(v8 + 16) = 0x100000000LL;
        *(_QWORD *)(v8 + 40) = v8 + 56;
        *(_QWORD *)(v8 + 32) = 0;
        *(_QWORD *)(v8 + 56) = v4;
        *(_QWORD *)(v8 + 48) = 0x200000001LL;
        v10 = *(unsigned int *)(v4 + 16);
        if ( (unsigned int)v10 >= *(_DWORD *)(v4 + 20) )
        {
          v37 = v9;
          sub_16CD150(v4 + 8, (const void *)(v4 + 24), 0, 8, v9, v38);
          v10 = *(unsigned int *)(v4 + 16);
          v9 = v37;
          v7 = (_QWORD *)v38;
        }
        *(_QWORD *)(*(_QWORD *)(v4 + 8) + 8 * v10) = v9;
        ++*(_DWORD *)(v4 + 16);
      }
      v11 = v7[5];
      v7[5] = v9;
      if ( v11 )
      {
        v12 = *(_QWORD *)(v11 + 40);
        if ( v12 != v11 + 56 )
        {
          v39 = v7;
          _libc_free(v12);
          v7 = v39;
        }
        v13 = *(_QWORD *)(v11 + 8);
        if ( v13 != v11 + 24 )
        {
          v40 = v7;
          _libc_free(v13);
          v7 = v40;
        }
        v41 = v7;
        j_j___libc_free_0(v11, 72);
        v7 = v41;
      }
    }
  }
  v42 = v7;
  v45 = v48;
  v46 = ".entry";
  v47 = 772;
  v14 = sub_22077B0(128);
  v15 = v14;
  if ( v14 )
    sub_1B90E50(v14, (__int64)&v45, v42);
  v16 = 0;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) )
  {
    v36 = sub_22077B0(48);
    v16 = (_QWORD *)v36;
    if ( v36 )
    {
      *(_QWORD *)(v36 + 8) = 0;
      *(_QWORD *)(v36 + 16) = 0;
      *(_BYTE *)(v36 + 24) = 4;
      *(_QWORD *)(v36 + 32) = 0;
      *(_QWORD *)(v36 + 40) = a2;
      *(_QWORD *)v36 = &unk_49F7008;
    }
  }
  v43 = v16;
  v45 = v48;
  v46 = ".continue";
  v47 = 772;
  v17 = sub_22077B0(128);
  v18 = v17;
  if ( v17 )
    sub_1B90E50(v17, (__int64)&v45, v43);
  v45 = v48;
  v46 = ".if";
  v47 = 772;
  v19 = sub_22077B0(128);
  v20 = v19;
  if ( v19 )
    sub_1B90E50(v19, (__int64)&v45, a3);
  v21 = sub_22077B0(136);
  v24 = v21;
  if ( v21 )
  {
    v25 = (_BYTE *)v48[0];
    *(_BYTE *)(v21 + 8) = 1;
    v26 = v48[1];
    *(_QWORD *)v21 = &unk_49F6D50;
    *(_QWORD *)(v21 + 16) = v21 + 32;
    sub_1B8E960((__int64 *)(v21 + 16), v25, (__int64)&v25[v26]);
    *(_QWORD *)(v24 + 48) = 0;
    *(_QWORD *)(v24 + 56) = v24 + 72;
    *(_QWORD *)(v24 + 64) = 0x100000000LL;
    *(_QWORD *)(v24 + 88) = 0x100000000LL;
    *(_QWORD *)(v24 + 80) = v24 + 96;
    *(_QWORD *)(v24 + 104) = 0;
    *(_QWORD *)v24 = &unk_49F7138;
    *(_QWORD *)(v24 + 112) = v15;
    *(_QWORD *)(v24 + 120) = v18;
    *(_BYTE *)(v24 + 128) = 1;
    *(_QWORD *)(v15 + 48) = v24;
    *(_QWORD *)(v18 + 48) = v24;
  }
  *(_QWORD *)(v15 + 104) = v4;
  v27 = *(unsigned int *)(v15 + 88);
  if ( (unsigned int)v27 >= *(_DWORD *)(v15 + 92) )
  {
    sub_16CD150(v15 + 80, (const void *)(v15 + 96), 0, 8, v22, v23);
    v27 = *(unsigned int *)(v15 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v15 + 80) + 8 * v27) = v20;
  v28 = (unsigned int)(*(_DWORD *)(v15 + 88) + 1);
  *(_DWORD *)(v15 + 88) = v28;
  if ( *(_DWORD *)(v15 + 92) <= (unsigned int)v28 )
  {
    sub_16CD150(v15 + 80, (const void *)(v15 + 96), 0, 8, v22, v23);
    v28 = *(unsigned int *)(v15 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v15 + 80) + 8 * v28) = v18;
  v29 = *(unsigned int *)(v20 + 64);
  ++*(_DWORD *)(v15 + 88);
  if ( (unsigned int)v29 >= *(_DWORD *)(v20 + 68) )
  {
    sub_16CD150(v20 + 56, (const void *)(v20 + 72), 0, 8, v22, v23);
    v29 = *(unsigned int *)(v20 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v20 + 56) + 8 * v29) = v15;
  v30 = *(unsigned int *)(v18 + 64);
  ++*(_DWORD *)(v20 + 64);
  if ( (unsigned int)v30 >= *(_DWORD *)(v18 + 68) )
  {
    sub_16CD150(v18 + 56, (const void *)(v18 + 72), 0, 8, v22, v23);
    v30 = *(unsigned int *)(v18 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v18 + 56) + 8 * v30) = v15;
  v31 = *(_QWORD *)(v15 + 48);
  ++*(_DWORD *)(v18 + 64);
  *(_QWORD *)(v20 + 48) = v31;
  *(_QWORD *)(v18 + 48) = v31;
  v32 = *(unsigned int *)(v20 + 88);
  if ( (unsigned int)v32 >= *(_DWORD *)(v20 + 92) )
  {
    sub_16CD150(v20 + 80, (const void *)(v20 + 96), 0, 8, v22, v23);
    v32 = *(unsigned int *)(v20 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v20 + 80) + 8 * v32) = v18;
  v33 = *(unsigned int *)(v18 + 64);
  ++*(_DWORD *)(v20 + 88);
  if ( (unsigned int)v33 >= *(_DWORD *)(v18 + 68) )
  {
    sub_16CD150(v18 + 56, (const void *)(v18 + 72), 0, 8, v22, v23);
    v33 = *(unsigned int *)(v18 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v18 + 56) + 8 * v33) = v20;
  v34 = (__int64 *)v48[0];
  ++*(_DWORD *)(v18 + 64);
  if ( v34 != &v49 )
    j_j___libc_free_0(v34, v49 + 1);
  return v24;
}
