// Function: sub_EFCCF0
// Address: 0xefccf0
//
__int64 __fastcall sub_EFCCF0(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r9
  int v13; // ecx
  int v14; // r15d
  __int64 v15; // r8
  unsigned int v16; // r14d
  unsigned int v17; // ecx
  int v18; // r12d
  unsigned int v19; // ecx
  unsigned int v20; // r12d
  unsigned int i; // r14d
  int v22; // eax
  int v23; // eax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  unsigned int v26; // eax
  unsigned int v27; // ecx
  int v28; // r12d
  __int64 v29; // r15
  __int64 result; // rax
  int v31; // r12d
  unsigned int v32; // ecx
  unsigned __int64 v33; // r8
  unsigned int v34; // r10d
  int v35; // edx
  unsigned int v36; // eax
  unsigned int v37; // eax
  int v38; // edx
  _QWORD *v39; // rdi
  __int64 v40; // rdx
  unsigned int v41; // eax
  unsigned int v42; // ecx
  _QWORD *v43; // rdi
  __int64 v44; // rdx
  int v45; // eax
  unsigned int v46; // edx
  unsigned int v47; // r10d
  unsigned int v48; // eax
  int v49; // edx
  unsigned int v50; // eax
  unsigned int v51; // eax
  unsigned int v52; // edx
  _QWORD *v53; // rdi
  __int64 v54; // rdx
  _QWORD *v55; // r12
  __int64 v56; // rdx
  _QWORD *v57; // r15
  __int64 v58; // rax
  unsigned int v59; // edx
  int v60; // eax
  _QWORD *v61; // r14
  __int64 v62; // rax
  int v63; // eax
  _QWORD *v64; // r12
  __int64 v65; // rax
  int v66; // eax
  __int64 v67; // [rsp+0h] [rbp-50h]
  unsigned __int64 v68; // [rsp+8h] [rbp-48h]
  int v69; // [rsp+8h] [rbp-48h]
  unsigned int v70; // [rsp+8h] [rbp-48h]
  unsigned int v71; // [rsp+14h] [rbp-3Ch]
  unsigned int v72; // [rsp+14h] [rbp-3Ch]
  unsigned int v73; // [rsp+14h] [rbp-3Ch]
  int v74; // [rsp+14h] [rbp-3Ch]
  int v75; // [rsp+14h] [rbp-3Ch]
  __int64 v76; // [rsp+18h] [rbp-38h]
  int v77; // [rsp+18h] [rbp-38h]
  unsigned int v78; // [rsp+18h] [rbp-38h]
  unsigned int v79; // [rsp+18h] [rbp-38h]
  unsigned int v80; // [rsp+18h] [rbp-38h]
  __int64 v81; // [rsp+18h] [rbp-38h]

  v8 = *(_DWORD *)(a3 + 12);
  *(_DWORD *)(a3 + 8) = 0;
  if ( v8 )
  {
    v9 = 0;
  }
  else
  {
    v81 = a5;
    sub_C8D5F0(a3, (const void *)(a3 + 16), 1u, 8u, a5, a6);
    a5 = v81;
    v9 = 8LL * *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + v9) = a1;
  v10 = *(_QWORD *)a3;
  v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v11;
  sub_EFCA60(a3, (char *)(v10 + 8 * v11), a4, a4 + a5);
  v13 = *(_DWORD *)(a2 + 48);
  v14 = *(_DWORD *)(a2 + 56);
  v15 = *(unsigned int *)(a3 + 8);
  v16 = (3 << v13) | *(_DWORD *)(a2 + 52);
  v17 = v14 + v13;
  *(_DWORD *)(a2 + 52) = v16;
  if ( v17 > 0x1F )
  {
    v64 = *(_QWORD **)(a2 + 24);
    v65 = v64[1];
    if ( (unsigned __int64)(v65 + 4) > v64[2] )
    {
      v78 = v15;
      sub_C8D290(*(_QWORD *)(a2 + 24), v64 + 3, v65 + 4, 1u, v15, v12);
      v65 = v64[1];
      v15 = v78;
    }
    *(_DWORD *)(*v64 + v65) = v16;
    v16 = 0;
    v64[1] += 4LL;
    v66 = *(_DWORD *)(a2 + 48);
    if ( v66 )
      v16 = 3u >> (32 - v66);
    v17 = ((_BYTE)v14 + (_BYTE)v66) & 0x1F;
  }
  *(_DWORD *)(a2 + 48) = v17;
  v18 = 3 << v17;
  v19 = v17 + 6;
  v20 = v16 | v18;
  *(_DWORD *)(a2 + 52) = v20;
  if ( v19 > 0x1F )
  {
    v61 = *(_QWORD **)(a2 + 24);
    v62 = v61[1];
    if ( (unsigned __int64)(v62 + 4) > v61[2] )
    {
      v79 = v15;
      sub_C8D290(*(_QWORD *)(a2 + 24), v61 + 3, v62 + 4, 1u, v15, v12);
      v62 = v61[1];
      v15 = v79;
    }
    *(_DWORD *)(*v61 + v62) = v20;
    v20 = 0;
    v61[1] += 4LL;
    v63 = *(_DWORD *)(a2 + 48);
    if ( v63 )
      v20 = 3u >> (32 - v63);
    *(_DWORD *)(a2 + 52) = v20;
    v19 = ((_BYTE)v63 + 6) & 0x1F;
  }
  *(_DWORD *)(a2 + 48) = v19;
  for ( i = v15; i > 0x1F; *(_DWORD *)(a2 + 48) = v19 )
  {
    v12 = i & 0x1F | 0x20;
    v23 = (i & 0x1F | 0x20) << v19;
    v19 += 6;
    v20 |= v23;
    *(_DWORD *)(a2 + 52) = v20;
    if ( v19 > 0x1F )
    {
      v24 = *(_QWORD **)(a2 + 24);
      v25 = v24[1];
      if ( (unsigned __int64)(v25 + 4) > v24[2] )
      {
        v71 = v15;
        v76 = *(_QWORD *)(a2 + 24);
        sub_C8D290(v76, v24 + 3, v25 + 4, 1u, v15, v12);
        v24 = (_QWORD *)v76;
        v12 = i & 0x1F | 0x20;
        v15 = v71;
        v25 = *(_QWORD *)(v76 + 8);
      }
      *(_DWORD *)(*v24 + v25) = v20;
      v20 = 0;
      v24[1] += 4LL;
      v22 = *(_DWORD *)(a2 + 48);
      if ( v22 )
        v20 = (unsigned int)v12 >> (32 - v22);
      v19 = ((_BYTE)v22 + 6) & 0x1F;
      *(_DWORD *)(a2 + 52) = v20;
    }
    i >>= 5;
  }
  v26 = i << v19;
  v27 = v19 + 6;
  v28 = v26 | v20;
  *(_DWORD *)(a2 + 52) = v28;
  if ( v27 > 0x1F )
  {
    v57 = *(_QWORD **)(a2 + 24);
    v58 = v57[1];
    if ( (unsigned __int64)(v58 + 4) > v57[2] )
    {
      v77 = v15;
      sub_C8D290(*(_QWORD *)(a2 + 24), v57 + 3, v58 + 4, 1u, v15, v12);
      v58 = v57[1];
      LODWORD(v15) = v77;
    }
    *(_DWORD *)(*v57 + v58) = v28;
    v59 = 0;
    v57[1] += 4LL;
    v60 = *(_DWORD *)(a2 + 48);
    if ( v60 )
      v59 = i >> (32 - v60);
    *(_DWORD *)(a2 + 52) = v59;
    *(_DWORD *)(a2 + 48) = ((_BYTE)v60 + 6) & 0x1F;
  }
  else
  {
    *(_DWORD *)(a2 + 48) = v27;
  }
  v29 = 0;
  result = 8LL * (unsigned int)v15;
  v67 = result;
  if ( (_DWORD)v15 )
  {
    do
    {
      v31 = *(_DWORD *)(a2 + 52);
      v32 = *(_DWORD *)(a2 + 48);
      v33 = *(_QWORD *)(*(_QWORD *)a3 + v29);
      v34 = v33;
      if ( v33 == (unsigned int)v33 )
      {
        if ( (unsigned int)v33 > 0x1F )
        {
          do
          {
            v51 = v34 & 0x1F | 0x20;
            v52 = v51 << v32;
            v32 += 6;
            v31 |= v52;
            *(_DWORD *)(a2 + 52) = v31;
            if ( v32 > 0x1F )
            {
              v53 = *(_QWORD **)(a2 + 24);
              v54 = v53[1];
              if ( (unsigned __int64)(v54 + 4) > v53[2] )
              {
                v70 = v34;
                v74 = v34 & 0x1F | 0x20;
                sub_C8D290((__int64)v53, v53 + 3, v54 + 4, 1u, v54 + 4, v12);
                v34 = v70;
                v51 = v74;
                v54 = v53[1];
              }
              v12 = 0;
              *(_DWORD *)(*v53 + v54) = v31;
              v53[1] += 4LL;
              v49 = *(_DWORD *)(a2 + 48);
              v50 = v51 >> (32 - v49);
              if ( v49 )
                v12 = v50;
              v32 = ((_BYTE)v49 + 6) & 0x1F;
              *(_DWORD *)(a2 + 52) = v12;
              v31 = v12;
            }
            v34 >>= 5;
            *(_DWORD *)(a2 + 48) = v32;
          }
          while ( v34 > 0x1F );
        }
        v48 = v34 << v32;
        v42 = v32 + 6;
        result = v31 | v48;
        *(_DWORD *)(a2 + 52) = result;
        if ( v42 > 0x1F )
        {
          v55 = *(_QWORD **)(a2 + 24);
          v56 = v55[1];
          if ( (unsigned __int64)(v56 + 4) > v55[2] )
          {
            v75 = result;
            v80 = v34;
            sub_C8D290(*(_QWORD *)(a2 + 24), v55 + 3, v56 + 4, 1u, v56 + 4, v12);
            v56 = v55[1];
            LODWORD(result) = v75;
            v34 = v80;
          }
          *(_DWORD *)(*v55 + v56) = result;
          v55[1] += 4LL;
          goto LABEL_32;
        }
      }
      else
      {
        if ( v33 > 0x1F )
        {
          do
          {
            v37 = v33 & 0x1F | 0x20;
            v38 = v37 << v32;
            v32 += 6;
            v31 |= v38;
            *(_DWORD *)(a2 + 52) = v31;
            if ( v32 > 0x1F )
            {
              v39 = *(_QWORD **)(a2 + 24);
              v40 = v39[1];
              if ( (unsigned __int64)(v40 + 4) > v39[2] )
              {
                v68 = v33;
                v72 = v33 & 0x1F | 0x20;
                sub_C8D290((__int64)v39, v39 + 3, v40 + 4, 1u, v33, v12);
                v33 = v68;
                v37 = v72;
                v40 = v39[1];
              }
              v12 = 0;
              *(_DWORD *)(*v39 + v40) = v31;
              v39[1] += 4LL;
              v35 = *(_DWORD *)(a2 + 48);
              v36 = v37 >> (32 - v35);
              if ( v35 )
                v12 = v36;
              v32 = ((_BYTE)v35 + 6) & 0x1F;
              *(_DWORD *)(a2 + 52) = v12;
              v31 = v12;
            }
            v33 >>= 5;
            *(_DWORD *)(a2 + 48) = v32;
          }
          while ( v33 > 0x1F );
          v34 = v33;
        }
        v41 = v34 << v32;
        v42 = v32 + 6;
        result = v31 | v41;
        *(_DWORD *)(a2 + 52) = result;
        if ( v42 > 0x1F )
        {
          v43 = *(_QWORD **)(a2 + 24);
          v44 = v43[1];
          if ( (unsigned __int64)(v44 + 4) > v43[2] )
          {
            v69 = result;
            v73 = v34;
            sub_C8D290((__int64)v43, v43 + 3, v44 + 4, 1u, v44 + 4, v12);
            LODWORD(result) = v69;
            v34 = v73;
            v44 = v43[1];
          }
          *(_DWORD *)(*v43 + v44) = result;
          v43[1] += 4LL;
LABEL_32:
          v45 = *(_DWORD *)(a2 + 48);
          v46 = 0;
          v47 = v34 >> (32 - v45);
          if ( v45 )
            v46 = v47;
          result = ((_BYTE)v45 + 6) & 0x1F;
          *(_DWORD *)(a2 + 52) = v46;
          *(_DWORD *)(a2 + 48) = result;
          goto LABEL_35;
        }
      }
      *(_DWORD *)(a2 + 48) = v42;
LABEL_35:
      v29 += 8;
    }
    while ( v67 != v29 );
  }
  return result;
}
