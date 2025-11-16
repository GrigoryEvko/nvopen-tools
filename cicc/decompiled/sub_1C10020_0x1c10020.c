// Function: sub_1C10020
// Address: 0x1c10020
//
__int64 __fastcall sub_1C10020(size_t a1, __int64 a2, __int64 a3)
{
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rax
  bool v8; // zf
  int v9; // eax
  __int64 v10; // r14
  int v11; // r13d
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // rdi
  __int64 v15; // r9
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 v20; // rcx
  __int64 v21; // r9
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r8
  unsigned int v25; // r12d
  __int64 v27; // rax
  int v28; // eax
  int v29; // r8d
  int v30; // r8d
  int v31; // r14d
  __int64 *v32; // r10
  int v33; // ecx
  int v34; // ebx
  __int64 *v35; // r10
  int v36; // ebx
  __int64 v37; // [rsp+10h] [rbp-F0h]
  __int64 v38; // [rsp+18h] [rbp-E8h] BYREF
  unsigned int v39; // [rsp+24h] [rbp-DCh] BYREF
  __int64 v40; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v41; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+38h] [rbp-C8h]
  __int64 v43; // [rsp+40h] [rbp-C0h]
  __int64 v44; // [rsp+48h] [rbp-B8h]
  __int64 v45; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+58h] [rbp-A8h]
  __int64 v47; // [rsp+60h] [rbp-A0h]
  __int64 v48; // [rsp+68h] [rbp-98h]
  __int64 *v49; // [rsp+70h] [rbp-90h] BYREF
  __int64 v50; // [rsp+78h] [rbp-88h]
  __int64 v51; // [rsp+80h] [rbp-80h]
  __int64 v52; // [rsp+88h] [rbp-78h]
  _BYTE *v53; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v54; // [rsp+A8h] [rbp-58h]
  _BYTE v55[80]; // [rsp+B0h] [rbp-50h] BYREF

  v38 = a2;
  v53 = v55;
  v54 = 0x400000000LL;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v41 = 1;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  sub_1353F00((__int64)&v41, 0);
  sub_1A97120((__int64)&v41, &v38, &v49);
  LODWORD(v43) = v43 + 1;
  if ( *v49 != -8 )
    --HIDWORD(v43);
  *v49 = v38;
  v7 = (unsigned int)v54;
  if ( (unsigned int)v54 >= HIDWORD(v54) )
  {
    sub_16CD150((__int64)&v53, v55, 0, 8, v5, v6);
    v7 = (unsigned int)v54;
  }
  *(_QWORD *)&v53[8 * v7] = v38;
  v39 = 0;
  v8 = (_DWORD)v54 == -1;
  v9 = v54 + 1;
  LODWORD(v54) = v54 + 1;
  if ( !v8 )
  {
    do
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)&v53[8 * v9 - 8];
        LODWORD(v54) = v9 - 1;
        if ( !(unsigned __int8)sub_1C07F10(a1, v10, &v49, a3) )
          break;
        v39 |= (unsigned int)v49;
        v9 = v54;
        if ( !(_DWORD)v54 )
          goto LABEL_11;
      }
      if ( *(_BYTE *)(v10 + 16) == 54 )
      {
        v27 = **(_QWORD **)(v10 - 24);
        if ( *(_BYTE *)(v27 + 8) == 16 )
          v27 = **(_QWORD **)(v27 + 16);
        v28 = *(_DWORD *)(v27 + 8) >> 8;
        if ( !v28 || v28 == 5 )
          v39 |= *(_DWORD *)(a1 + 128);
      }
      sub_1C0CC70((_QWORD *)a1, v10, &v39, (__int64)&v41, (__int64)&v53, (__int64)&v45);
      v9 = v54;
    }
    while ( (_DWORD)v54 );
LABEL_11:
    v11 = v39;
    if ( v39 )
      goto LABEL_13;
  }
  v11 = v47;
  if ( !(_DWORD)v47 )
  {
LABEL_13:
    v12 = v38;
    v13 = *(_DWORD *)(a1 + 32);
    v14 = a1 + 8;
    v40 = v38;
    if ( v13 )
    {
      v15 = *(_QWORD *)(a1 + 16);
      v16 = (v13 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v17 = (__int64 *)(v15 + 24LL * v16);
      v18 = *v17;
      if ( v38 == *v17 )
      {
LABEL_15:
        v17[1] = a3;
        v19 = *(_DWORD *)(a1 + 32);
        if ( v19 )
          goto LABEL_16;
        goto LABEL_33;
      }
      v31 = 1;
      v32 = 0;
      while ( v18 != -8 )
      {
        if ( !v32 && v18 == -16 )
          v32 = v17;
        v16 = (v13 - 1) & (v31 + v16);
        v17 = (__int64 *)(v15 + 24LL * v16);
        v18 = *v17;
        if ( v38 == *v17 )
          goto LABEL_15;
        ++v31;
      }
      v33 = *(_DWORD *)(a1 + 24);
      if ( v32 )
        v17 = v32;
      ++*(_QWORD *)(a1 + 8);
      v29 = v33 + 1;
      if ( 4 * (v33 + 1) < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(a1 + 28) - v29 > v13 >> 3 )
        {
LABEL_30:
          *(_DWORD *)(a1 + 24) = v29;
          if ( *v17 != -8 )
            --*(_DWORD *)(a1 + 28);
          v17[1] = 0;
          *v17 = v12;
          *((_DWORD *)v17 + 4) = 0;
          v17[1] = a3;
          v19 = *(_DWORD *)(a1 + 32);
          if ( v19 )
          {
LABEL_16:
            v20 = v40;
            v21 = *(_QWORD *)(a1 + 16);
            v22 = (v19 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
            v23 = (__int64 *)(v21 + 24LL * v22);
            v24 = *v23;
            if ( *v23 == v40 )
            {
LABEL_17:
              *((_DWORD *)v23 + 4) = v11;
              sub_1C0AE50((__int64)&v49, a1 + 40, &v40);
              v25 = v39;
              goto LABEL_18;
            }
            v34 = 1;
            v35 = 0;
            while ( v24 != -8 )
            {
              if ( v24 == -16 && !v35 )
                v35 = v23;
              v22 = (v19 - 1) & (v34 + v22);
              v23 = (__int64 *)(v21 + 24LL * v22);
              v24 = *v23;
              if ( v40 == *v23 )
                goto LABEL_17;
              ++v34;
            }
            v36 = *(_DWORD *)(a1 + 24);
            if ( v35 )
              v23 = v35;
            ++*(_QWORD *)(a1 + 8);
            v30 = v36 + 1;
            if ( 4 * (v36 + 1) < 3 * v19 )
            {
              if ( v19 - *(_DWORD *)(a1 + 28) - v30 > v19 >> 3 )
                goto LABEL_49;
              goto LABEL_35;
            }
LABEL_34:
            v19 *= 2;
LABEL_35:
            sub_1C0AC80(v14, v19);
            sub_1C09A10(v14, &v40, &v49);
            v23 = v49;
            v20 = v40;
            v30 = *(_DWORD *)(a1 + 24) + 1;
LABEL_49:
            *(_DWORD *)(a1 + 24) = v30;
            if ( *v23 != -8 )
              --*(_DWORD *)(a1 + 28);
            *v23 = v20;
            v23[1] = 0;
            *((_DWORD *)v23 + 4) = 0;
            goto LABEL_17;
          }
LABEL_33:
          ++*(_QWORD *)(a1 + 8);
          goto LABEL_34;
        }
        v37 = a1 + 8;
        sub_1C0AC80(v14, v13);
LABEL_29:
        sub_1C09A10(v37, &v40, &v49);
        v17 = v49;
        v12 = v40;
        v29 = *(_DWORD *)(a1 + 24) + 1;
        v14 = v37;
        goto LABEL_30;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
    }
    v37 = a1 + 8;
    sub_1C0AC80(v14, 2 * v13);
    goto LABEL_29;
  }
  v51 = 0;
  v49 = 0;
  v50 = 0;
  v52 = 0;
  v39 = sub_1C0EF50(a1, 0, (__int64)&v45, a3, (__int64)&v49);
  sub_1C0AFA0(a1, v38, v39, (_DWORD)v51 == 0, a3);
  v25 = v39;
  j___libc_free_0(v50);
LABEL_18:
  j___libc_free_0(v46);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  j___libc_free_0(v42);
  return v25;
}
