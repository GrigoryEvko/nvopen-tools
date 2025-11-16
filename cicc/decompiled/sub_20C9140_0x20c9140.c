// Function: sub_20C9140
// Address: 0x20c9140
//
__int64 __fastcall sub_20C9140(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v5; // r15d
  __int64 v6; // rax
  int v7; // eax
  int v8; // r8d
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 (*v12)(); // rax
  __int64 v13; // r13
  __int64 v14; // r14
  unsigned int v15; // r12d
  __int64 v16; // rbx
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // r10
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  _BYTE *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // r13
  __int64 *v28; // rbx
  __int64 v29; // rdx
  __int64 *v30; // r13
  __int64 *v31; // rbx
  __int64 v32; // rdx
  __int64 *v33; // r13
  __int64 *v34; // rbx
  __int64 v35; // rdx
  _BYTE *v36; // rbx
  int v37; // esi
  __int64 v38; // rdx
  __int64 v39; // [rsp+0h] [rbp-330h]
  __int64 v40; // [rsp+8h] [rbp-328h]
  __int64 v41; // [rsp+10h] [rbp-320h]
  unsigned int v42; // [rsp+18h] [rbp-318h]
  __int64 v43; // [rsp+18h] [rbp-318h]
  __int64 *v44; // [rsp+40h] [rbp-2F0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-2E8h]
  _BYTE v46[128]; // [rsp+50h] [rbp-2E0h] BYREF
  __int64 *v47; // [rsp+D0h] [rbp-260h] BYREF
  __int64 v48; // [rsp+D8h] [rbp-258h]
  _BYTE v49[128]; // [rsp+E0h] [rbp-250h] BYREF
  __int64 *v50; // [rsp+160h] [rbp-1D0h] BYREF
  __int64 v51; // [rsp+168h] [rbp-1C8h]
  _BYTE v52[128]; // [rsp+170h] [rbp-1C0h] BYREF
  _BYTE *v53; // [rsp+1F0h] [rbp-140h] BYREF
  __int64 v54; // [rsp+1F8h] [rbp-138h]
  _BYTE v55[304]; // [rsp+200h] [rbp-130h] BYREF

  v2 = a1;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( !*(_BYTE *)(a2 + 522) )
    return v2;
  v5 = *(_DWORD *)(*(_QWORD *)(a2 + 328) + 48LL);
  v6 = sub_15E38F0(*(_QWORD *)a2);
  v7 = sub_14DD7D0(v6);
  v10 = *(_QWORD *)(a2 + 16);
  v11 = 0;
  v42 = v7 - 7;
  v12 = *(__int64 (**)())(*(_QWORD *)v10 + 40LL);
  if ( v12 != sub_1D00B00 )
    v11 = ((__int64 (__fastcall *)(__int64, _QWORD))v12)(v10, 0);
  v13 = *(_QWORD *)(a2 + 328);
  v14 = a2 + 320;
  v47 = (__int64 *)v49;
  v44 = (__int64 *)v46;
  v50 = (__int64 *)v52;
  v45 = 0x1000000000LL;
  v48 = 0x1000000000LL;
  v51 = 0x1000000000LL;
  v53 = v55;
  v54 = 0x1000000000LL;
  if ( v13 != a2 + 320 )
  {
    v41 = v2;
    v15 = v42;
    v43 = a2;
    v16 = v13;
    do
    {
      if ( *(_BYTE *)(v16 + 182) )
      {
        v23 = (unsigned int)v45;
        if ( (unsigned int)v45 >= HIDWORD(v45) )
        {
          sub_16CD150((__int64)&v44, v46, 0, 8, v8, v9);
          v23 = (unsigned int)v45;
        }
        v44[v23] = v16;
        LODWORD(v45) = v45 + 1;
      }
      else if ( v15 <= 1 && *(_BYTE *)(v16 + 180) )
      {
        v25 = (unsigned int)v51;
        if ( (unsigned int)v51 >= HIDWORD(v51) )
        {
          sub_16CD150((__int64)&v50, v52, 0, 8, v8, v9);
          v25 = (unsigned int)v51;
        }
        v50[v25] = v16;
        LODWORD(v51) = v51 + 1;
      }
      else if ( *(_QWORD *)(v16 + 72) == *(_QWORD *)(v16 + 64) )
      {
        v26 = (unsigned int)v48;
        if ( (unsigned int)v48 >= HIDWORD(v48) )
        {
          sub_16CD150((__int64)&v47, v49, 0, 8, v8, v9);
          v26 = (unsigned int)v48;
        }
        v47[v26] = v16;
        LODWORD(v48) = v48 + 1;
      }
      v17 = sub_1DD5EE0(v16);
      if ( v17 != v16 + 24 && *(_DWORD *)(v11 + 44) == **(unsigned __int16 **)(v17 + 16) )
      {
        v18 = *(_QWORD *)(v17 + 32);
        v19 = v5;
        v9 = *(_QWORD *)(v18 + 24);
        if ( v15 > 1 )
          v19 = *(_DWORD *)(*(_QWORD *)(v18 + 64) + 48LL);
        v20 = v19;
        v21 = (unsigned int)v54;
        if ( (unsigned int)v54 >= HIDWORD(v54) )
        {
          v39 = v20;
          v40 = *(_QWORD *)(v18 + 24);
          sub_16CD150((__int64)&v53, v55, 0, 16, v8, v9);
          v21 = (unsigned int)v54;
          v20 = v39;
          v9 = v40;
        }
        v22 = &v53[16 * v21];
        *v22 = v9;
        v22[1] = v20;
        LODWORD(v54) = v54 + 1;
      }
      v16 = *(_QWORD *)(v16 + 8);
    }
    while ( v14 != v16 );
    v2 = v41;
    if ( (_DWORD)v45 )
    {
      sub_20C8D30(v41, v5, *(_QWORD *)(v43 + 328));
      v27 = v47;
      v28 = &v47[(unsigned int)v48];
      if ( v47 != v28 )
      {
        do
        {
          v29 = *v27++;
          sub_20C8D30(v41, v5, v29);
        }
        while ( v28 != v27 );
      }
      v30 = v44;
      v31 = &v44[(unsigned int)v45];
      if ( v31 != v44 )
      {
        do
        {
          v32 = *v30++;
          sub_20C8D30(v41, *(_DWORD *)(v32 + 48), v32);
        }
        while ( v31 != v30 );
      }
      v33 = v50;
      v34 = &v50[(unsigned int)v51];
      if ( v50 != v34 )
      {
        do
        {
          v35 = *v33++;
          sub_20C8D30(v41, v5, v35);
        }
        while ( v34 != v33 );
      }
      v36 = v53;
      v24 = &v53[16 * (unsigned int)v54];
      if ( v53 == v24 )
        goto LABEL_24;
      do
      {
        v37 = *((_DWORD *)v36 + 2);
        v38 = *(_QWORD *)v36;
        v36 += 16;
        sub_20C8D30(v41, v37, v38);
      }
      while ( v24 != v36 );
    }
    v24 = v53;
LABEL_24:
    if ( v24 != v55 )
      _libc_free((unsigned __int64)v24);
  }
  if ( v50 != (__int64 *)v52 )
    _libc_free((unsigned __int64)v50);
  if ( v47 != (__int64 *)v49 )
    _libc_free((unsigned __int64)v47);
  if ( v44 != (__int64 *)v46 )
    _libc_free((unsigned __int64)v44);
  return v2;
}
