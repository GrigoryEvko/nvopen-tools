// Function: sub_2D35D10
// Address: 0x2d35d10
//
void __fastcall sub_2D35D10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 *v12; // rcx
  unsigned __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // r10d
  __int64 v23; // r9
  __int64 v24; // rdx
  unsigned int *v25; // rax
  __int64 v26; // rax
  int v27; // ecx
  unsigned int v28; // esi
  __int64 v29; // rdx
  unsigned int *v30; // rax
  unsigned int v31; // r8d
  unsigned int v32; // r10d
  unsigned int v33; // r9d
  __int64 i; // rax
  __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // [rsp+20h] [rbp-170h]
  unsigned int v39; // [rsp+28h] [rbp-168h]
  unsigned int v40; // [rsp+28h] [rbp-168h]
  unsigned int v41; // [rsp+2Ch] [rbp-164h]
  unsigned int v42; // [rsp+2Ch] [rbp-164h]
  unsigned int v43; // [rsp+30h] [rbp-160h]
  unsigned int v44; // [rsp+30h] [rbp-160h]
  __int64 v45; // [rsp+30h] [rbp-160h]
  unsigned __int64 v46; // [rsp+38h] [rbp-158h]
  __int64 v47; // [rsp+40h] [rbp-150h] BYREF
  _BYTE *v48; // [rsp+48h] [rbp-148h] BYREF
  __int64 v49; // [rsp+50h] [rbp-140h]
  _BYTE v50[72]; // [rsp+58h] [rbp-138h] BYREF
  __int64 v51; // [rsp+A0h] [rbp-F0h] BYREF
  _BYTE *v52; // [rsp+A8h] [rbp-E8h]
  __int64 v53; // [rsp+B0h] [rbp-E0h]
  _BYTE v54[72]; // [rsp+B8h] [rbp-D8h] BYREF
  __int64 v55; // [rsp+100h] [rbp-90h] BYREF
  _BYTE *v56; // [rsp+108h] [rbp-88h]
  __int64 v57; // [rsp+110h] [rbp-80h]
  _BYTE v58[120]; // [rsp+118h] [rbp-78h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
  v38 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a2 + 8) + 8LL;
  if ( !*(_DWORD *)(a1 + 24) )
    return;
  v8 = 0;
  while ( 2 )
  {
    if ( v6 )
    {
      v9 = *(_DWORD *)(v7 - 8);
      *(_DWORD *)v6 = v9;
    }
    else
    {
      v9 = MEMORY[0];
    }
    if ( v9 > 0xFFFFFFFD )
      goto LABEL_3;
    v10 = *(_QWORD *)(v7 + 200);
    *(_QWORD *)(v6 + 8) = 0;
    *(_QWORD *)(v6 + 208) = v10;
    *(_QWORD *)(v6 + 192) = 0;
    *(_DWORD *)(v6 + 200) = 0;
    *(_DWORD *)(v6 + 204) = 0;
    memset(
      (void *)((v6 + 16) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((_DWORD)v6 + 8 - (((_DWORD)v6 + 16) & 0xFFFFFFF8) + 192) >> 3));
    *(_QWORD *)(v6 + 208) = *(_QWORD *)(v7 + 200);
    v48 = v50;
    v49 = 0x400000000LL;
    v47 = v7;
    sub_2D29C80((__int64)&v47, 0, a3, 0, a5, a6);
    v15 = *(_DWORD *)(v47 + 192);
    if ( v15 )
    {
      v11 = (unsigned int)v49;
      for ( i = (unsigned int)(v49 - 1); v15 > (unsigned int)i; LODWORD(v49) = v49 + 1 )
      {
        v35 = (__int64)v48;
        v36 = v11 + 1;
        v13 = *(_QWORD *)(*(_QWORD *)&v48[16 * i] + 8LL * *(unsigned int *)&v48[16 * i + 12]) & 0xFFFFFFFFFFFFFFC0LL;
        v37 = (*(_QWORD *)(*(_QWORD *)&v48[16 * i] + 8LL * *(unsigned int *)&v48[16 * i + 12]) & 0x3FLL) + 1;
        if ( v36 > HIDWORD(v49) )
        {
          v45 = v37;
          v46 = v13;
          sub_C8D5F0((__int64)&v48, v50, v36, 0x10u, v13, v14);
          v35 = (__int64)v48;
          v37 = v45;
          v13 = v46;
        }
        v12 = (unsigned __int64 *)(16LL * (unsigned int)v49 + v35);
        *v12 = v13;
        v12[1] = v37;
        i = (unsigned int)v49;
        v11 = (unsigned int)(v49 + 1);
      }
    }
    v51 = v7;
    v52 = v54;
    v53 = 0x400000000LL;
    sub_2D29C80((__int64)&v51, *(_DWORD *)(v7 + 196), v11, (__int64)v12, v13, v14);
    v16 = (unsigned int)v49;
    while ( 1 )
    {
      a3 = (unsigned int)v53;
      if ( !(_DWORD)v16 || *((_DWORD *)v48 + 3) >= *((_DWORD *)v48 + 2) )
        break;
      v17 = (__int64)&v48[16 * v16 - 16];
      v18 = (unsigned __int64)&v52[16 * (unsigned int)v53 - 16];
      v19 = *(unsigned int *)(v17 + 12);
      a3 = *(_QWORD *)v17;
      if ( (_DWORD)v19 == *(_DWORD *)(v18 + 12) && *(_QWORD *)v18 == a3 )
        goto LABEL_24;
LABEL_13:
      v20 = *(_DWORD *)(v6 + 200);
      v21 = *(unsigned int *)(a3 + 8 * v19 + 4);
      v22 = *(_DWORD *)(a3 + 8 * v19);
      v23 = *(unsigned int *)(a3 + 4 * v19 + 128);
      if ( v20 )
      {
        v40 = *(_DWORD *)(a3 + 4 * v19 + 128);
        v42 = *(_DWORD *)(a3 + 8 * v19 + 4);
        v44 = *(_DWORD *)(a3 + 8 * v19);
        v55 = v6 + 8;
        v56 = v58;
        v57 = 0x400000000LL;
        sub_2D2BC70((__int64)&v55, v22, a3, v17, v21, v23);
        v32 = v44;
        v31 = v42;
        v33 = v40;
      }
      else
      {
        v24 = *(unsigned int *)(v6 + 204);
        if ( (_DWORD)v24 != 16 )
        {
          if ( (_DWORD)v24 )
          {
            v25 = (unsigned int *)(v6 + 12);
            do
            {
              if ( v22 < *v25 )
                break;
              ++v20;
              v25 += 2;
            }
            while ( (_DWORD)v24 != v20 );
          }
          LODWORD(v55) = v20;
          *(_DWORD *)(v6 + 204) = sub_2D28A50(v6 + 8, (unsigned int *)&v55, v24, v22, v21, v23);
          goto LABEL_20;
        }
        v55 = v6 + 8;
        v57 = 0x400000000LL;
        v30 = (unsigned int *)(v6 + 12);
        v56 = v58;
        do
        {
          if ( v22 < *v30 )
            break;
          ++v20;
          v30 += 2;
        }
        while ( v20 != 16 );
        v39 = v21;
        v41 = v22;
        v43 = v23;
        sub_2D29C80((__int64)&v55, v20, v24, v17, v21, v23);
        v31 = v39;
        v32 = v41;
        v33 = v43;
      }
      sub_2D35090((__int64)&v55, v32, v31, v33);
      if ( v56 != v58 )
        _libc_free((unsigned __int64)v56);
LABEL_20:
      v26 = (__int64)&v48[16 * (unsigned int)v49 - 16];
      v27 = *(_DWORD *)(v26 + 12) + 1;
      *(_DWORD *)(v26 + 12) = v27;
      v16 = (unsigned int)v49;
      if ( v27 == *(_DWORD *)&v48[16 * (unsigned int)v49 - 8] )
      {
        v28 = *(_DWORD *)(v47 + 192);
        if ( v28 )
        {
          sub_F03D40((__int64 *)&v48, v28);
          v16 = (unsigned int)v49;
        }
      }
    }
    if ( (_DWORD)v53 )
    {
      v17 = *((unsigned int *)v52 + 2);
      if ( *((_DWORD *)v52 + 3) < (unsigned int)v17 )
      {
        v29 = (__int64)&v48[16 * v16 - 16];
        v19 = *(unsigned int *)(v29 + 12);
        a3 = *(_QWORD *)v29;
        goto LABEL_13;
      }
    }
LABEL_24:
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
    if ( v48 != v50 )
      _libc_free((unsigned __int64)v48);
LABEL_3:
    ++v8;
    v6 += 216;
    v7 += 216;
    if ( v38 != v8 )
      continue;
    break;
  }
}
