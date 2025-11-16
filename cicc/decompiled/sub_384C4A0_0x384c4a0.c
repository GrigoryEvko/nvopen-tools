// Function: sub_384C4A0
// Address: 0x384c4a0
//
__int64 __fastcall sub_384C4A0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // r14
  unsigned __int64 v10; // rcx
  __int64 v11; // rdx
  _BYTE *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  bool v21; // zf
  int v22; // eax
  __int64 v23; // rbx
  char v24; // al
  __int64 v25; // r8
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // r14
  __int64 *v30; // r13
  __int64 v31; // r15
  __int64 v32; // rbx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  _BYTE *v35; // rbx
  __int64 v36; // r15
  __int64 v37; // r14
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 v43; // rax
  _DWORD *v44; // rsi
  _BYTE *v45; // rax
  __int64 v46; // [rsp+0h] [rbp-100h]
  const void *v47; // [rsp+10h] [rbp-F0h]
  _BYTE *v48; // [rsp+28h] [rbp-D8h]
  _BYTE *v49; // [rsp+28h] [rbp-D8h]
  _BYTE *v50; // [rsp+30h] [rbp-D0h]
  __int64 v51; // [rsp+40h] [rbp-C0h]
  char v52; // [rsp+4Eh] [rbp-B2h]
  char v53; // [rsp+4Fh] [rbp-B1h]
  _BYTE *v54; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+58h] [rbp-A8h]
  _BYTE v56[32]; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE *v57; // [rsp+80h] [rbp-80h] BYREF
  __int64 v58; // [rsp+88h] [rbp-78h]
  _BYTE v59[112]; // [rsp+90h] [rbp-70h] BYREF

  v8 = (__int64)(a3[13] - a3[12]) >> 3;
  v9 = (unsigned int)v8;
  if ( *(_DWORD *)(a2 + 12) < (unsigned int)v8 )
  {
    v44 = (_DWORD *)(a2 + 16);
    *(v44 - 2) = 0;
    sub_C8D5F0(a2, v44, (unsigned int)v8, 0x10u, a5, a6);
    v45 = *(_BYTE **)a2;
    do
    {
      if ( v45 )
      {
        *v45 = 0;
        *((_DWORD *)v45 + 1) = 0;
        *((_DWORD *)v45 + 2) = 0;
        *((_DWORD *)v45 + 3) = 0;
      }
      v45 += 16;
      --v9;
    }
    while ( v9 );
  }
  else
  {
    v10 = *(unsigned int *)(a2 + 8);
    v11 = v10;
    if ( (unsigned int)v8 <= v10 )
      v11 = (unsigned int)v8;
    if ( v11 )
    {
      v12 = *(_BYTE **)a2;
      v11 = *(_QWORD *)a2 + 16 * v11;
      do
      {
        *v12 = 0;
        v12 += 16;
        *((_DWORD *)v12 - 3) = 0;
        *((_DWORD *)v12 - 2) = 0;
        *((_DWORD *)v12 - 1) = 0;
      }
      while ( (_BYTE *)v11 != v12 );
      v10 = *(unsigned int *)(a2 + 8);
    }
    if ( (unsigned int)v8 > v10 )
    {
      v43 = *(_QWORD *)a2 + 16 * v10;
      v11 = (unsigned int)v8 - v10;
      if ( (unsigned int)v8 != v10 )
      {
        do
        {
          if ( v43 )
          {
            *(_BYTE *)v43 = 0;
            *(_DWORD *)(v43 + 4) = 0;
            *(_DWORD *)(v43 + 8) = 0;
            *(_DWORD *)(v43 + 12) = 0;
          }
          v43 += 16;
          --v11;
        }
        while ( v11 );
      }
    }
  }
  *(_DWORD *)(a2 + 8) = v8;
  v13 = a3[41];
  v57 = v59;
  v58 = 0x800000000LL;
  sub_384C060((__int64)&v57, v13, v11, v10, a5, a6);
  v54 = v56;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v55 = 0x400000000LL;
  v16 = (unsigned int)v58;
  *(_QWORD *)a1 = a1 + 16;
  v47 = (const void *)(a1 + 16);
  v48 = v57;
  v50 = &v57[8 * v16];
  if ( v50 == v57 )
  {
    *(_DWORD *)(a2 + 8) = 0;
  }
  else
  {
    do
    {
      v17 = *((_QWORD *)v50 - 1);
      v18 = 16LL * *(unsigned int *)(v17 + 24);
      *(_BYTE *)(*(_QWORD *)a2 + v18) = 1;
      *(_DWORD *)(*(_QWORD *)a2 + v18 + 8) = *(_DWORD *)(*(_QWORD *)a2 + v18 + 4);
      v19 = (unsigned int)v55;
      v20 = (unsigned int)v55 + 1LL;
      if ( v20 > HIDWORD(v55) )
      {
        sub_C8D5F0((__int64)&v54, v56, v20, 8u, v14, v15);
        v19 = (unsigned int)v55;
      }
      *(_QWORD *)&v54[8 * v19] = v17;
      v21 = (_DWORD)v55 == -1;
      v22 = v55 + 1;
      LODWORD(v55) = v55 + 1;
      if ( !v21 )
      {
        v53 = 1;
        do
        {
          v23 = *(_QWORD *)&v54[8 * v22 - 8];
          LODWORD(v55) = v22 - 1;
          v24 = sub_384C020((_QWORD *)a2, v23);
          v26 = v51;
          LOBYTE(v26) = v53;
          v52 = v24;
          BYTE1(v26) = v24;
          v27 = *(unsigned int *)(a1 + 8);
          v51 = v26;
          if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            sub_C8D5F0(a1, v47, v27 + 1, 0x10u, v25, v15);
            v27 = *(unsigned int *)(a1 + 8);
          }
          v28 = (__int64 *)(*(_QWORD *)a1 + 16 * v27);
          *v28 = v23;
          v28[1] = v51;
          ++*(_DWORD *)(a1 + 8);
          v14 = *(_QWORD *)(v23 + 112);
          v29 = (__int64 *)v14;
          v30 = (__int64 *)(v14 + 8LL * *(unsigned int *)(v23 + 120));
          if ( (__int64 *)v14 != v30 )
          {
            do
            {
              while ( 1 )
              {
                v31 = *v29;
                v32 = *(unsigned int *)(*v29 + 24);
                if ( !(unsigned __int8)sub_384C020((_QWORD *)a2, *v29) )
                {
                  if ( v53 )
                    ++*(_DWORD *)(*(_QWORD *)a2 + 16LL * (unsigned int)v32 + 4);
                  if ( v52 )
                    ++*(_DWORD *)(*(_QWORD *)a2 + 16 * v32 + 12);
                  if ( (unsigned __int8)sub_384C020((_QWORD *)a2, v31) )
                    break;
                }
                if ( v30 == ++v29 )
                  goto LABEL_28;
              }
              v33 = (unsigned int)v55;
              v34 = (unsigned int)v55 + 1LL;
              if ( v34 > HIDWORD(v55) )
              {
                sub_C8D5F0((__int64)&v54, v56, v34, 8u, v14, v15);
                v33 = (unsigned int)v55;
              }
              ++v29;
              *(_QWORD *)&v54[8 * v33] = v31;
              LODWORD(v55) = v55 + 1;
            }
            while ( v30 != v29 );
          }
LABEL_28:
          v22 = v55;
          v53 = 0;
        }
        while ( (_DWORD)v55 );
      }
      v50 -= 8;
    }
    while ( v48 != v50 );
    v35 = &v57[8 * (unsigned int)v58];
    v49 = v57;
    if ( v57 != v35 )
    {
      v36 = v46;
      do
      {
        while ( 1 )
        {
          v37 = *((_QWORD *)v35 - 1);
          if ( !(unsigned __int8)sub_384C020((_QWORD *)a2, v37) )
            break;
          v35 -= 8;
          if ( v49 == v35 )
            goto LABEL_37;
        }
        v40 = *(unsigned int *)(a1 + 8);
        LOWORD(v36) = 256;
        if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, v47, v40 + 1, 0x10u, v38, v39);
          v40 = *(unsigned int *)(a1 + 8);
        }
        v41 = (__int64 *)(*(_QWORD *)a1 + 16 * v40);
        v35 -= 8;
        *v41 = v37;
        v41[1] = v36;
        ++*(_DWORD *)(a1 + 8);
      }
      while ( v49 != v35 );
    }
LABEL_37:
    *(_DWORD *)(a2 + 8) = 0;
    if ( v54 != v56 )
      _libc_free((unsigned __int64)v54);
    v48 = v57;
  }
  if ( v48 != v59 )
    _libc_free((unsigned __int64)v48);
  return a1;
}
