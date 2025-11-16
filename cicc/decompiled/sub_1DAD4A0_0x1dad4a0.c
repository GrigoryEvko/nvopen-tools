// Function: sub_1DAD4A0
// Address: 0x1dad4a0
//
void __fastcall sub_1DAD4A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 *v18; // rbx
  unsigned int v19; // esi
  bool v20; // bl
  unsigned __int64 v21; // rdx
  __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // r13
  unsigned __int64 v25; // r14
  __int64 v26; // r12
  __int64 v27; // rdx
  unsigned int v28; // edx
  _QWORD *v29; // rdi
  __int64 v30; // r12
  __int64 v31; // r14
  __int64 *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned int v36; // ecx
  __int64 v37; // rsi
  unsigned int v38; // [rsp+0h] [rbp-C0h]
  __int64 v39; // [rsp+8h] [rbp-B8h]
  __int64 v40; // [rsp+10h] [rbp-B0h]
  __int64 v41; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v42; // [rsp+10h] [rbp-B0h]
  __int64 v44; // [rsp+20h] [rbp-A0h]
  __int64 v45; // [rsp+28h] [rbp-98h]
  __int64 v46; // [rsp+28h] [rbp-98h]
  __int64 v47; // [rsp+28h] [rbp-98h]
  __int64 v48; // [rsp+28h] [rbp-98h]
  __int64 v49; // [rsp+30h] [rbp-90h] BYREF
  _QWORD *v50; // [rsp+38h] [rbp-88h] BYREF
  __int64 v51; // [rsp+40h] [rbp-80h]
  _QWORD v52[15]; // [rsp+48h] [rbp-78h] BYREF

  v44 = a2;
  v45 = *(_QWORD *)(a7 + 272);
  v12 = a1 + 216;
  v13 = *(unsigned int *)(sub_1DA9310(v45, a2) + 48);
  v49 = v12;
  v15 = *(unsigned int *)(a1 + 296);
  v16 = *(_QWORD *)(*(_QWORD *)(v45 + 392) + 16 * v13 + 8);
  v50 = v52;
  v51 = 0x400000000LL;
  if ( (_DWORD)v15 )
  {
    v40 = v16;
    sub_1DAAC30((__int64)&v49, a2, v15, v14, v16, (int)v52);
    v16 = v40;
    v20 = a5 != 0 && a4 != 0;
    if ( !v20 )
    {
LABEL_7:
      v20 = 1;
      v21 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      v22 = (a2 >> 1) & 3;
      v23 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      LODWORD(v24) = v22;
      goto LABEL_8;
    }
  }
  else
  {
    v17 = *(unsigned int *)(a1 + 300);
    if ( (_DWORD)v17 )
    {
      v18 = (__int64 *)(a1 + 224);
      v19 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
      do
      {
        if ( (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) > v19 )
          break;
        v15 = (unsigned int)(v15 + 1);
        v18 += 2;
      }
      while ( (_DWORD)v17 != (_DWORD)v15 );
    }
    v52[0] = v12;
    LODWORD(v51) = 1;
    v52[1] = v17 | (v15 << 32);
    v20 = a5 != 0 && a4 != 0;
    if ( !v20 )
      goto LABEL_7;
  }
  v41 = v16;
  v32 = (__int64 *)sub_1DB3C70(a4, a2);
  if ( v32 == (__int64 *)(*(_QWORD *)a4 + 24LL * *(unsigned int *)(a4 + 8))
    || (v21 = a2 & 0xFFFFFFFFFFFFFFF8LL,
        v22 = (a2 >> 1) & 3,
        v23 = a2 & 0xFFFFFFFFFFFFFFF8LL,
        LODWORD(v24) = v22,
        (*(_DWORD *)((*v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v32 >> 1) & 3) > ((unsigned int)v22
                                                                                           | *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)))
    || (v16 = v41, v32[2] != a5) )
  {
    if ( !a6 )
    {
LABEL_18:
      v29 = v50;
      goto LABEL_19;
    }
    v33 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v33 >= *(_DWORD *)(a6 + 12) )
    {
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v16, (int)v52);
      v33 = *(unsigned int *)(a6 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a6 + 8 * v33) = a2;
    v29 = v50;
    ++*(_DWORD *)(a6 + 8);
    goto LABEL_19;
  }
  if ( (*(_DWORD *)((v32[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v32[1] >> 1) & 3) < (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(v41 >> 1)
                                                                                              & 3) )
  {
    v16 = v32[1];
    v20 = 0;
  }
LABEL_8:
  if ( !(_DWORD)v51 )
    goto LABEL_9;
  v29 = v50;
  if ( *((_DWORD *)v50 + 3) >= *((_DWORD *)v50 + 2) )
    goto LABEL_9;
  v46 = (unsigned int)v51;
  v30 = (__int64)&v50[2 * (unsigned int)v51 - 2];
  v31 = *(_QWORD *)v30;
  v38 = *(_DWORD *)(v30 + 12);
  v39 = 16LL * v38;
  if ( (*(_DWORD *)((*(_QWORD *)(v31 + v39) & 0xFFFFFFFFFFFFFFF8LL) + 24)
      | (unsigned int)(*(__int64 *)(v31 + v39) >> 1) & 3) > ((unsigned int)v24 | *(_DWORD *)(v21 + 24)) )
  {
    v25 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    v26 = (v16 >> 1) & 3;
    goto LABEL_25;
  }
  if ( v22 == 3 )
    v44 = *(_QWORD *)(v21 + 8) & 0xFFFFFFFFFFFFFFF9LL;
  else
    v44 = v21 | (2 * v22 + 2);
  v34 = v49;
  if ( *(_DWORD *)(v49 + 80) )
    v35 = v31 + 4LL * v38 + 144;
  else
    v35 = v31 + 4LL * v38 + 64;
  if ( ((a3 ^ *(_DWORD *)v35) & 0x7FFFFFFF) == 0
    && ((*(_BYTE *)(v35 + 3) ^ HIBYTE(a3)) & 0x80u) == 0
    && *(_QWORD *)(v31 + v39 + 8) == v44 )
  {
    *(_DWORD *)(v30 + 12) = v38 + 1;
    v36 = v51;
    if ( v38 + 1 == LODWORD(v50[2 * (unsigned int)v51 - 1]) )
    {
      v37 = *(unsigned int *)(v34 + 80);
      if ( (_DWORD)v37 )
      {
        v48 = v16;
        sub_39460A0(&v50, v37);
        v36 = v51;
        v16 = v48;
      }
    }
    v23 = v44 & 0xFFFFFFFFFFFFFFF8LL;
    v24 = (v44 >> 1) & 3;
    if ( v36 )
    {
      v29 = v50;
      v25 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      v26 = (v16 >> 1) & 3;
      if ( *((_DWORD *)v50 + 2) <= *((_DWORD *)v50 + 3) )
        goto LABEL_10;
      v46 = v36;
LABEL_25:
      v28 = *(_DWORD *)((*(_QWORD *)(v29[2 * v46 - 2] + 16LL * HIDWORD(v29[2 * v46 - 1])) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (*(__int64 *)(v29[2 * v46 - 2] + 16LL * HIDWORD(v29[2 * v46 - 1])) >> 1) & 3;
      if ( v28 < ((unsigned int)v26 | *(_DWORD *)(v25 + 24)) )
      {
        v16 = *(_QWORD *)(v29[2 * v46 - 2] + 16LL * HIDWORD(v29[2 * v46 - 1]));
LABEL_16:
        if ( (*(_DWORD *)(v23 + 24) | (unsigned int)v24) < v28 )
          sub_1DAD0A0((__int64)&v49, v44, v16, a3);
        goto LABEL_18;
      }
LABEL_10:
      if ( !v20 && a6 )
      {
        v27 = *(unsigned int *)(a6 + 8);
        if ( (unsigned int)v27 >= *(_DWORD *)(a6 + 12) )
        {
          v42 = v23;
          v47 = v16;
          sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v16, (int)v52);
          v27 = *(unsigned int *)(a6 + 8);
          v23 = v42;
          v16 = v47;
        }
        *(_QWORD *)(*(_QWORD *)a6 + 8 * v27) = v16;
        ++*(_DWORD *)(a6 + 8);
      }
      v28 = v26 | *(_DWORD *)(v25 + 24);
      goto LABEL_16;
    }
LABEL_9:
    v25 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    v26 = (v16 >> 1) & 3;
    goto LABEL_10;
  }
LABEL_19:
  if ( v29 != v52 )
    _libc_free((unsigned __int64)v29);
}
