// Function: sub_A1A630
// Address: 0xa1a630
//
__int64 __fastcall sub_A1A630(__int64 a1, int a2, __int64 *a3)
{
  __int64 v5; // r14
  __int64 v6; // r13
  int v7; // r15d
  unsigned __int64 v8; // rsi
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rax
  int v13; // ecx
  __int64 v14; // r8
  unsigned int v15; // r9d
  unsigned int v16; // eax
  unsigned int v17; // ecx
  int v18; // eax
  unsigned __int8 v19; // al
  unsigned __int64 v20; // r13
  unsigned int v21; // esi
  int v22; // ecx
  unsigned int v23; // r9d
  int v24; // r8d
  unsigned int v25; // ecx
  int v26; // r8d
  _QWORD *v27; // rdi
  __int64 v28; // rdx
  unsigned int v29; // ecx
  int v30; // edx
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rsi
  int v39; // ecx
  unsigned int v40; // eax
  unsigned int v41; // r9d
  unsigned int v42; // ecx
  int v43; // r9d
  unsigned __int8 v44; // al
  _QWORD *v45; // r13
  __int64 v46; // rdx
  unsigned int v47; // edx
  int v48; // eax
  _QWORD *v49; // rdi
  __int64 v50; // rdx
  unsigned int v51; // ecx
  int v52; // edx
  __int64 v53; // rsi
  __int64 v55; // [rsp+10h] [rbp-70h]
  int v56; // [rsp+18h] [rbp-68h]
  int v57; // [rsp+1Ch] [rbp-64h]
  int v58; // [rsp+1Ch] [rbp-64h]
  unsigned int v59; // [rsp+1Ch] [rbp-64h]
  unsigned int v60; // [rsp+20h] [rbp-60h]
  __int64 v61; // [rsp+20h] [rbp-60h]
  __int64 v62; // [rsp+28h] [rbp-58h]
  _BYTE *v63; // [rsp+30h] [rbp-50h] BYREF
  __int64 v64; // [rsp+38h] [rbp-48h]
  _BYTE v65[64]; // [rsp+40h] [rbp-40h] BYREF

  if ( a2 != *(_DWORD *)(a1 + 60) )
  {
    v5 = 0;
    v63 = v65;
    v64 = 0x200000000LL;
    sub_9C8C60((__int64)&v63, a2);
    v6 = (unsigned int)v64;
    sub_A17B10(a1, 3u, *(_DWORD *)(a1 + 56));
    v7 = v6;
    sub_A17CC0(a1, 1u, 6);
    v8 = (unsigned int)v6;
    sub_A17CC0(a1, v6, 6);
    v9 = 4 * v6;
    if ( v7 )
    {
      do
      {
        v8 = *(unsigned int *)&v63[v5];
        v5 += 4;
        sub_A17DE0(a1, v8, 6);
      }
      while ( v9 != v5 );
    }
    *(_DWORD *)(a1 + 60) = a2;
    if ( v63 != v65 )
      _libc_free(v63, v8);
  }
  v10 = 0;
  v11 = *a3;
  sub_A17B10(a1, 2u, *(_DWORD *)(a1 + 56));
  sub_A17CC0(a1, *(_DWORD *)(v11 + 8), 5);
  v12 = *(unsigned int *)(v11 + 8);
  v55 = 16 * v12;
  if ( (_DWORD)v12 )
  {
    while ( 1 )
    {
      v13 = *(_DWORD *)(a1 + 48);
      v14 = v10 + *(_QWORD *)v11;
      v15 = *(_BYTE *)(v14 + 8) & 1;
      v16 = v15 << v13;
      v17 = v13 + 1;
      v18 = *(_DWORD *)(a1 + 52) | v16;
      *(_DWORD *)(a1 + 52) = v18;
      if ( v17 > 0x1F )
      {
        v45 = *(_QWORD **)(a1 + 24);
        v46 = v45[1];
        if ( (unsigned __int64)(v46 + 4) > v45[2] )
        {
          v58 = v18;
          v60 = v15;
          v62 = v14;
          sub_C8D290(*(_QWORD *)(a1 + 24), v45 + 3, v46 + 4, 1);
          v46 = v45[1];
          v18 = v58;
          v15 = v60;
          v14 = v62;
        }
        *(_DWORD *)(*v45 + v46) = v18;
        v47 = 0;
        v45[1] += 4LL;
        v48 = *(_DWORD *)(a1 + 48);
        if ( v48 )
          v47 = v15 >> (32 - v48);
        *(_DWORD *)(a1 + 52) = v47;
        *(_DWORD *)(a1 + 48) = ((_BYTE)v48 + 1) & 0x1F;
      }
      else
      {
        *(_DWORD *)(a1 + 48) = v17;
      }
      v19 = *(_BYTE *)(v14 + 8);
      if ( (v19 & 1) == 0 )
        break;
      v20 = *(_QWORD *)v14;
      v21 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 == v21 )
      {
        sub_A17CC0(a1, v21, 8);
LABEL_35:
        v10 += 16;
        if ( v55 == v10 )
          goto LABEL_22;
      }
      else
      {
        if ( v20 > 0x7F )
        {
          do
          {
            while ( 1 )
            {
              v22 = *(_DWORD *)(a1 + 48);
              v23 = v20 & 0x7F | 0x80;
              v24 = v23 << v22;
              v25 = v22 + 8;
              v26 = *(_DWORD *)(a1 + 52) | v24;
              *(_DWORD *)(a1 + 52) = v26;
              if ( v25 > 0x1F )
                break;
              v20 >>= 7;
              *(_DWORD *)(a1 + 48) = v25;
              if ( v20 <= 0x7F )
                goto LABEL_20;
            }
            v27 = *(_QWORD **)(a1 + 24);
            v28 = v27[1];
            if ( (unsigned __int64)(v28 + 4) > v27[2] )
            {
              v57 = v26;
              sub_C8D290(v27, v27 + 3, v28 + 4, 1);
              v26 = v57;
              v23 = v20 & 0x7F | 0x80;
              v28 = v27[1];
            }
            *(_DWORD *)(*v27 + v28) = v26;
            v29 = 0;
            v27[1] += 4LL;
            v30 = *(_DWORD *)(a1 + 48);
            if ( v30 )
              v29 = v23 >> (32 - v30);
            v20 >>= 7;
            *(_DWORD *)(a1 + 52) = v29;
            *(_DWORD *)(a1 + 48) = ((_BYTE)v30 + 8) & 0x1F;
          }
          while ( v20 > 0x7F );
LABEL_20:
          v21 = v20;
        }
        v10 += 16;
        sub_A17B10(a1, v21, 8);
        if ( v55 == v10 )
          goto LABEL_22;
      }
    }
    v39 = *(_DWORD *)(a1 + 48);
    v40 = (v19 >> 1) & 7;
    v41 = v40 << v39;
    v42 = v39 + 3;
    v43 = *(_DWORD *)(a1 + 52) | v41;
    *(_DWORD *)(a1 + 52) = v43;
    if ( v42 > 0x1F )
    {
      v49 = *(_QWORD **)(a1 + 24);
      v50 = v49[1];
      if ( (unsigned __int64)(v50 + 4) > v49[2] )
      {
        v56 = v43;
        v59 = v40;
        v61 = v14;
        sub_C8D290(v49, v49 + 3, v50 + 4, 1);
        v43 = v56;
        v40 = v59;
        v14 = v61;
        v50 = v49[1];
      }
      *(_DWORD *)(*v49 + v50) = v43;
      v51 = 0;
      v49[1] += 4LL;
      v52 = *(_DWORD *)(a1 + 48);
      if ( v52 )
        v51 = v40 >> (32 - v52);
      *(_DWORD *)(a1 + 52) = v51;
      *(_DWORD *)(a1 + 48) = ((_BYTE)v52 + 3) & 0x1F;
    }
    else
    {
      *(_DWORD *)(a1 + 48) = v42;
    }
    v44 = (*(_BYTE *)(v14 + 8) >> 1) & 7;
    if ( v44 > 2u )
    {
      if ( ((v44 + 5) & 7u) > 2 )
LABEL_33:
        sub_C64ED0("Invalid encoding", 1);
    }
    else
    {
      if ( !v44 )
        goto LABEL_33;
      sub_A17DE0(a1, *(_QWORD *)v14, 5);
    }
    goto LABEL_35;
  }
LABEL_22:
  v31 = *(_QWORD *)(a1 + 136);
  v32 = *(_QWORD *)(a1 + 128);
  if ( v31 == v32 )
  {
    if ( v31 != *(_QWORD *)(a1 + 144) )
    {
      if ( v31 )
      {
LABEL_55:
        *(_OWORD *)v31 = 0;
        *(_OWORD *)(v31 + 16) = 0;
        v31 = *(_QWORD *)(a1 + 136);
      }
      v53 = v31 + 32;
      *(_QWORD *)(a1 + 136) = v53;
      goto LABEL_57;
    }
  }
  else
  {
    v33 = v31 - 32;
    if ( a2 == *(_DWORD *)(v31 - 32) )
      goto LABEL_24;
    do
    {
      v33 = v32;
      if ( a2 == *(_DWORD *)v32 )
      {
        v34 = *(_QWORD *)(v32 + 16);
        if ( v34 != *(_QWORD *)(v32 + 24) )
          goto LABEL_25;
        goto LABEL_40;
      }
      v32 += 32;
    }
    while ( v31 != v32 );
    if ( v31 != *(_QWORD *)(a1 + 144) )
      goto LABEL_55;
  }
  sub_A18EF0((__int64 *)(a1 + 128), (char *)v31);
  v53 = *(_QWORD *)(a1 + 136);
LABEL_57:
  *(_DWORD *)(v53 - 32) = a2;
  v33 = *(_QWORD *)(a1 + 136) - 32LL;
LABEL_24:
  v34 = *(_QWORD *)(v33 + 16);
  if ( v34 == *(_QWORD *)(v33 + 24) )
  {
LABEL_40:
    sub_A1A390((char **)(v33 + 8), (char *)v34, a3);
    v37 = *(_QWORD *)(v33 + 16);
  }
  else
  {
LABEL_25:
    if ( v34 )
    {
      v35 = *a3;
      *(_QWORD *)(v34 + 8) = 0;
      *a3 = 0;
      *(_QWORD *)v34 = v35;
      v36 = a3[1];
      a3[1] = 0;
      *(_QWORD *)(v34 + 8) = v36;
      v34 = *(_QWORD *)(v33 + 16);
    }
    v37 = v34 + 16;
    *(_QWORD *)(v33 + 16) = v37;
  }
  return (unsigned int)((v37 - *(_QWORD *)(v33 + 8)) >> 4) + 3;
}
