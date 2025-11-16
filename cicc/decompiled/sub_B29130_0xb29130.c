// Function: sub_B29130
// Address: 0xb29130
//
__int64 __fastcall sub_B29130(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // r13
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rax
  unsigned int v10; // eax
  _QWORD *v11; // r12
  _BYTE *v12; // rsi
  unsigned int v13; // r13d
  unsigned int i; // eax
  __int64 v15; // rdx
  __int64 v16; // r14
  int v17; // r10d
  __int64 v18; // rax
  int v19; // r10d
  __int64 v20; // rdx
  int v21; // edx
  char *v22; // r10
  __int64 *v23; // r14
  _QWORD *v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rcx
  unsigned int v27; // edx
  __int64 v28; // rdx
  unsigned __int64 v29; // r9
  _QWORD *v30; // rdx
  __int64 v31; // r12
  __int64 v32; // rbx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 *v36; // r13
  __int64 *v37; // rbx
  __int64 v38; // rdx
  unsigned int v39; // eax
  _QWORD *v40; // r14
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned int v43; // eax
  __int64 v44; // rsi
  _BYTE *v45; // rbx
  __int64 result; // rax
  _BYTE *v47; // r12
  _BYTE *v48; // rdi
  _QWORD *v49; // [rsp+8h] [rbp-1508h]
  char *v50; // [rsp+10h] [rbp-1500h]
  __int64 *v51; // [rsp+18h] [rbp-14F8h]
  unsigned int v52; // [rsp+38h] [rbp-14D8h]
  int v53; // [rsp+38h] [rbp-14D8h]
  unsigned int v54; // [rsp+3Ch] [rbp-14D4h]
  int v55; // [rsp+48h] [rbp-14C8h]
  unsigned __int64 v56; // [rsp+48h] [rbp-14C8h]
  __int64 v57; // [rsp+48h] [rbp-14C8h]
  __int64 *v58; // [rsp+50h] [rbp-14C0h] BYREF
  int v59; // [rsp+58h] [rbp-14B8h]
  char v60; // [rsp+60h] [rbp-14B0h] BYREF
  _BYTE *v61; // [rsp+A0h] [rbp-1470h] BYREF
  unsigned int v62; // [rsp+A8h] [rbp-1468h]
  unsigned int v63; // [rsp+ACh] [rbp-1464h]
  _BYTE v64[1024]; // [rsp+B0h] [rbp-1460h] BYREF
  _QWORD v65[2]; // [rsp+4B0h] [rbp-1060h] BYREF
  _QWORD v66[64]; // [rsp+4C0h] [rbp-1050h] BYREF
  _BYTE *v67; // [rsp+6C0h] [rbp-E50h]
  __int64 v68; // [rsp+6C8h] [rbp-E48h]
  _BYTE v69[3584]; // [rsp+6D0h] [rbp-E40h] BYREF
  __int64 v70; // [rsp+14D0h] [rbp-40h]

  v5 = sub_B197A0(a1, a3, a4);
  v6 = (__int64 *)v5;
  if ( v5 )
  {
    v7 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
    v8 = *(_DWORD *)(v5 + 44) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  if ( v8 >= *(_DWORD *)(a1 + 56) )
    BUG();
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v7);
  v51 = *(__int64 **)(v9 + 8);
  if ( !v51 )
    return sub_B28E70(a1, a2);
  v10 = *(_DWORD *)(v9 + 16);
  v11 = v65;
  v58 = v6;
  v54 = v10;
  v65[0] = v66;
  v65[1] = 0x4000000001LL;
  v67 = v69;
  v68 = 0x4000000000LL;
  v70 = a2;
  v66[0] = 0;
  v59 = 0;
  sub_B1C510(&v61, &v58, 1);
  v12 = v6;
  v13 = 0;
  *(_DWORD *)(sub_B20CA0((__int64)v65, (__int64)v12) + 4) = 0;
  for ( i = v62; v62; i = v62 )
  {
    while ( 1 )
    {
      v15 = (__int64)&v61[16 * i - 16];
      v16 = *(_QWORD *)v15;
      v17 = *(_DWORD *)(v15 + 8);
      v62 = i - 1;
      v12 = (_BYTE *)v16;
      v55 = v17;
      v18 = sub_B20CA0((__int64)v11, v16);
      v19 = v55;
      v20 = *(unsigned int *)(v18 + 32);
      if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 36) )
      {
        v12 = (_BYTE *)(v18 + 40);
        v53 = v55;
        v57 = v18;
        sub_C8D5F0(v18 + 24, v18 + 40, v20 + 1, 4);
        v18 = v57;
        v19 = v53;
        v20 = *(unsigned int *)(v57 + 32);
      }
      *(_DWORD *)(*(_QWORD *)(v18 + 24) + 4 * v20) = v19;
      v21 = *(_DWORD *)v18;
      ++*(_DWORD *)(v18 + 32);
      if ( !v21 )
      {
        ++v13;
        *(_DWORD *)(v18 + 4) = v19;
        *(_DWORD *)(v18 + 12) = v13;
        *(_DWORD *)(v18 + 8) = v13;
        *(_DWORD *)v18 = v13;
        sub_B1A4E0((__int64)v11, v16);
        v12 = (_BYTE *)v16;
        sub_B1CE50((__int64)&v58, v16, v70);
        v22 = (char *)&v58[v59];
        if ( v58 != (__int64 *)v22 )
        {
          v23 = v58;
          v24 = v11;
          v25 = v13;
          do
          {
            v31 = *v23;
            if ( *v23 )
            {
              v26 = (unsigned int)(*(_DWORD *)(v31 + 44) + 1);
              v27 = *(_DWORD *)(v31 + 44) + 1;
            }
            else
            {
              v27 = 0;
              v26 = 0;
            }
            if ( v27 >= *(_DWORD *)(a1 + 56) )
              BUG();
            if ( v54 < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v26) + 16LL) )
            {
              v28 = v62;
              v12 = (_BYTE *)0xFFFFFFFF00000000LL;
              v29 = v4 & 0xFFFFFFFF00000000LL | v25;
              v4 = v29;
              if ( (unsigned __int64)v62 + 1 > v63 )
              {
                v12 = v64;
                v49 = v24;
                v50 = v22;
                v52 = v25;
                v56 = v29;
                sub_C8D5F0(&v61, v64, v62 + 1LL, 16);
                v28 = v62;
                v24 = v49;
                v22 = v50;
                v25 = v52;
                v29 = v56;
              }
              v30 = &v61[16 * v28];
              *v30 = v31;
              v30[1] = v29;
              ++v62;
            }
            ++v23;
          }
          while ( v22 != (char *)v23 );
          v22 = (char *)v58;
          v13 = v25;
          v11 = v24;
        }
        if ( v22 != &v60 )
          break;
      }
      i = v62;
      if ( !v62 )
        goto LABEL_25;
    }
    _libc_free(v22, v12);
  }
LABEL_25:
  if ( v61 != v64 )
    _libc_free(v61, v12);
  sub_B20E50((__int64)v11);
  v32 = *v51;
  v33 = 1;
  *(_QWORD *)(sub_B20CA0((__int64)v11, *(_QWORD *)(v65[0] + 8LL)) + 16) = v32;
  v34 = sub_B1B2D0(v11, 1);
  v36 = v35;
  v37 = (__int64 *)v34;
  if ( (__int64 *)v34 != v35 )
  {
    do
    {
      v44 = *v37;
      if ( *v37 )
      {
        v38 = (unsigned int)(*(_DWORD *)(v44 + 44) + 1);
        v39 = *(_DWORD *)(v44 + 44) + 1;
      }
      else
      {
        v38 = 0;
        v39 = 0;
      }
      v40 = 0;
      if ( v39 < *(_DWORD *)(a1 + 56) )
        v40 = *(_QWORD **)(*(_QWORD *)(a1 + 48) + 8 * v38);
      v41 = *(_QWORD *)(sub_B20CA0((__int64)v11, v44) + 16);
      if ( v41 )
      {
        v42 = (unsigned int)(*(_DWORD *)(v41 + 44) + 1);
        v43 = *(_DWORD *)(v41 + 44) + 1;
      }
      else
      {
        v42 = 0;
        v43 = 0;
      }
      v33 = 0;
      if ( v43 < *(_DWORD *)(a1 + 56) )
        v33 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v42);
      ++v37;
      sub_B1AE50(v40, v33);
    }
    while ( v36 != v37 );
  }
  v45 = v67;
  result = 7LL * (unsigned int)v68;
  v47 = &v67[56 * (unsigned int)v68];
  if ( v67 != v47 )
  {
    do
    {
      v47 -= 56;
      v48 = (_BYTE *)*((_QWORD *)v47 + 3);
      result = (__int64)(v47 + 40);
      if ( v48 != v47 + 40 )
        result = _libc_free(v48, v33);
    }
    while ( v45 != v47 );
    v47 = v67;
  }
  if ( v47 != v69 )
    result = _libc_free(v47, v33);
  if ( (_QWORD *)v65[0] != v66 )
    return _libc_free(v65[0], v33);
  return result;
}
