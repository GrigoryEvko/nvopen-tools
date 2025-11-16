// Function: sub_285A420
// Address: 0x285a420
//
__int64 __fastcall sub_285A420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  __int16 v8; // r14
  __int64 v9; // rdi
  unsigned int v10; // r13d
  unsigned int v11; // ecx
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v16; // rdi
  unsigned int v17; // r13d
  int v18; // eax
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rsi
  unsigned int v24; // edx
  __int64 v25; // r12
  __int64 v26; // r12
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  _QWORD *v29; // r9
  __int64 v30; // r13
  __int64 v31; // rsi
  _QWORD *v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned __int64 v35; // rax
  _QWORD *v36; // r14
  _QWORD *v37; // r12
  int v38; // r15d
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  int v42; // r13d
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 *v46; // r12
  __int64 v47; // r15
  unsigned int v48; // eax
  __int64 v49; // r8
  __int64 v50; // r9
  unsigned __int64 v51; // rcx
  _BOOL8 v52; // rax
  __int64 v53; // r14
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  _QWORD *v56; // r14
  _QWORD *v57; // r12
  int v58; // r15d
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rax
  __int64 v62[3]; // [rsp+10h] [rbp-50h] BYREF
  char v63; // [rsp+28h] [rbp-38h] BYREF

  v8 = *(_WORD *)(a2 + 24);
  switch ( v8 )
  {
    case 0:
      v9 = *(_QWORD *)(a2 + 32);
      v10 = *(_DWORD *)(v9 + 32);
      v11 = v10 - 1;
      v12 = 1LL << ((unsigned __int8)v10 - 1);
      v13 = *(_QWORD *)(v9 + 24);
      if ( v10 > 0x40 )
      {
        v16 = v9 + 24;
        v17 = v10 + 1;
        if ( (*(_QWORD *)(v13 + 8LL * (v11 >> 6)) & v12) != 0 )
        {
          v18 = sub_C44500(v16);
          v19 = v17;
          v6 = 0;
          if ( v19 - v18 > 0x40 )
            return v6;
          goto LABEL_10;
        }
        v14 = v17 - sub_C444A0(v16);
      }
      else
      {
        if ( (v12 & v13) == 0 )
        {
          if ( v13 )
          {
            _BitScanReverse64(&v13, v13);
            v14 = 65 - (v13 ^ 0x3F);
            goto LABEL_6;
          }
LABEL_10:
          v20 = *(unsigned int *)(a1 + 8);
          if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            sub_C8D5F0(a1, (const void *)(a1 + 16), v20 + 1, 8u, a5, a6);
            v20 = *(unsigned int *)(a1 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a1 + 8 * v20) = 17;
          v21 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v21;
          v22 = *(_QWORD *)(a2 + 32);
          v23 = *(__int64 **)(v22 + 24);
          v24 = *(_DWORD *)(v22 + 32);
          if ( v24 > 0x40 )
          {
            v25 = *v23;
          }
          else
          {
            v25 = 0;
            if ( v24 )
              v25 = (__int64)((_QWORD)v23 << (64 - (unsigned __int8)v24)) >> (64 - (unsigned __int8)v24);
          }
          if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            sub_C8D5F0(a1, (const void *)(a1 + 16), v21 + 1, 8u, a5, a6);
            v21 = *(unsigned int *)(a1 + 8);
          }
          v6 = 1;
          *(_QWORD *)(*(_QWORD *)a1 + 8 * v21) = v25;
          ++*(_DWORD *)(a1 + 8);
          return v6;
        }
        if ( !v10 )
          goto LABEL_10;
        v35 = ~(v13 << (64 - (unsigned __int8)v10));
        if ( v35 )
        {
          _BitScanReverse64(&v35, v35);
          v14 = v10 + 1 - (v35 ^ 0x3F);
        }
        else
        {
          v14 = v10 - 63;
        }
      }
LABEL_6:
      v6 = 0;
      if ( v14 > 0x40 )
        return v6;
      goto LABEL_10;
    case 15:
      v26 = *(_QWORD *)(a2 - 8);
      v6 = 0;
      if ( v26 )
      {
        v27 = *(unsigned int *)(a1 + 8);
        v28 = *(unsigned int *)(a1 + 12);
        v62[0] = *(_QWORD *)(a2 - 8);
        if ( v27 + 1 > v28 )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v27 + 1, 8u, a5, a6);
          v27 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v27) = 4101;
        v29 = *(_QWORD **)(a1 + 64);
        v30 = *(unsigned int *)(a1 + 72);
        ++*(_DWORD *)(a1 + 8);
        v31 = (__int64)&v29[v30];
        v32 = sub_284FE40(v29, v31, v62);
        if ( (_QWORD *)v31 == v32 )
        {
          if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
          {
            sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v30 + 1, 8u, v33, v34);
            v32 = (_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * *(unsigned int *)(a1 + 72));
          }
          *v32 = v26;
          v33 = *(unsigned int *)(a1 + 8);
          ++*(_DWORD *)(a1 + 72);
        }
        else
        {
          v30 = (unsigned int)(((__int64)v32 - v34) >> 3);
        }
        if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v33 + 1, 8u, v33, v34);
          v33 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v33) = v30;
        v6 = 1;
        ++*(_DWORD *)(a1 + 8);
      }
      return v6;
    case 6:
      v36 = *(_QWORD **)(a2 + 32);
      v37 = &v36[*(_QWORD *)(a2 + 40)];
      if ( v36 != v37 )
      {
        v38 = 0;
        v6 = sub_285A420(a1, *v36);
        while ( 1 )
        {
          ++v36;
          ++v38;
          if ( v37 == v36 )
            break;
          while ( 1 )
          {
            v6 &= sub_285A420(a1, *v36);
            if ( !v38 )
              break;
            v41 = *(unsigned int *)(a1 + 8);
            if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
            {
              sub_C8D5F0(a1, (const void *)(a1 + 16), v41 + 1, 8u, v39, v40);
              v41 = *(unsigned int *)(a1 + 8);
            }
            ++v36;
            ++v38;
            *(_QWORD *)(*(_QWORD *)a1 + 8 * v41) = 30;
            ++*(_DWORD *)(a1 + 8);
            if ( v37 == v36 )
              return v6;
          }
        }
        return v6;
      }
      return 1;
    case 7:
      v42 = sub_285A420(a1, *(_QWORD *)(a2 + 32));
      v6 = sub_285A420(a1, *(_QWORD *)(a2 + 40)) & v42;
      v45 = *(unsigned int *)(a1 + 8);
      if ( v45 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v45 + 1, 8u, v43, v44);
        v45 = *(unsigned int *)(a1 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v45) = 27;
      ++*(_DWORD *)(a1 + 8);
      return v6;
  }
  LOBYTE(v6) = v8 == 14 || (unsigned __int16)(v8 - 2) <= 2u;
  if ( (_BYTE)v6 )
  {
    v46 = v62;
    v47 = *(_DWORD *)(*(_QWORD *)(a2 + 40) + 8LL) >> 8;
    v48 = sub_285A420(a1, *(_QWORD *)(a2 + 32));
    v51 = *(unsigned int *)(a1 + 12);
    v62[1] = v47;
    v6 = v48;
    v52 = v8 != 4;
    v62[0] = 4097;
    v53 = 4097;
    v62[2] = 2 * v52 + 5;
    v54 = *(unsigned int *)(a1 + 8);
    v55 = v54 + 1;
    if ( v54 + 1 > v51 )
      goto LABEL_49;
    while ( 1 )
    {
      ++v46;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v54) = v53;
      v54 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v54;
      if ( v46 == (__int64 *)&v63 )
        break;
      v55 = v54 + 1;
      v53 = *v46;
      if ( v54 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
LABEL_49:
        sub_C8D5F0(a1, (const void *)(a1 + 16), v55, 8u, v49, v50);
        v54 = *(unsigned int *)(a1 + 8);
      }
    }
  }
  else
  {
    if ( v8 != 5 )
      return v6;
    v56 = *(_QWORD **)(a2 + 32);
    v57 = &v56[*(_QWORD *)(a2 + 40)];
    if ( v56 == v57 )
      return 1;
    v58 = 0;
    v6 = sub_285A420(a1, *v56);
    while ( 1 )
    {
      ++v56;
      ++v58;
      if ( v57 == v56 )
        break;
      while ( 1 )
      {
        v6 &= sub_285A420(a1, *v56);
        if ( !v58 )
          break;
        v61 = *(unsigned int *)(a1 + 8);
        if ( v61 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v61 + 1, 8u, v59, v60);
          v61 = *(unsigned int *)(a1 + 8);
        }
        ++v56;
        ++v58;
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v61) = 34;
        ++*(_DWORD *)(a1 + 8);
        if ( v57 == v56 )
          return v6;
      }
    }
  }
  return v6;
}
