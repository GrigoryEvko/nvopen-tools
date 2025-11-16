// Function: sub_394C0B0
// Address: 0x394c0b0
//
__int64 __fastcall sub_394C0B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int v5; // ebx
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r10
  unsigned int v10; // ebx
  unsigned int v11; // esi
  unsigned int v12; // eax
  unsigned __int64 v13; // rax
  unsigned int v14; // edx
  unsigned int v15; // ecx
  unsigned __int64 v16; // rax
  int v17; // eax
  unsigned int v18; // r15d
  int v19; // ecx
  __int64 v20; // r12
  __int64 v21; // rdi
  char v22; // al
  unsigned int v23; // r8d
  unsigned __int64 v24; // rsi
  unsigned int v25; // ebx
  __int64 v26; // rax
  int v27; // eax
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rax
  unsigned int v30; // ebx
  unsigned int v31; // eax
  int v32; // eax
  __int64 *v34; // rsi
  __int64 *v35; // rcx
  __int64 *v36; // rbx
  __int64 *v37; // rax
  __int64 v38; // rdi
  __int64 *v39; // rdx
  __int64 *v40; // rsi
  unsigned int v41; // edi
  __int64 *v42; // rdx
  __int64 v43; // rax
  _QWORD *v44; // rbx
  __int64 v45; // rax
  _QWORD *v46; // r15
  __int64 v47; // r14
  __int64 v48; // r14
  __int64 v49; // r14
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 v52; // rbx
  int v53; // eax
  unsigned int v54; // [rsp+0h] [rbp-60h]
  __int64 v55; // [rsp+0h] [rbp-60h]
  unsigned int v56; // [rsp+0h] [rbp-60h]
  __int64 v57; // [rsp+0h] [rbp-60h]
  __int64 v58; // [rsp+0h] [rbp-60h]
  unsigned int v60; // [rsp+8h] [rbp-58h]
  unsigned int v61; // [rsp+8h] [rbp-58h]
  unsigned __int64 v62; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v63; // [rsp+18h] [rbp-48h]
  unsigned __int64 v64; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v65; // [rsp+28h] [rbp-38h]

  v4 = a2;
  v5 = *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8;
  v6 = v5 - (*(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 8);
  v7 = sub_15F2050(*(_QWORD *)(a1 + 8));
  v8 = sub_1632FA0(v7);
  v63 = v5;
  v9 = v8;
  if ( v5 > 0x40 )
  {
    v55 = v8;
    sub_16A4EF0((__int64)&v62, 0, 0);
    v65 = v5;
    sub_16A4EF0((__int64)&v64, 0, 0);
    v9 = v55;
  }
  else
  {
    v62 = 0;
    v65 = v5;
    v64 = 0;
  }
  sub_14BB090(a2, (__int64)&v62, v9, 0, 0, 0, 0, 0);
  v10 = v63;
  v11 = v63;
  if ( v63 > 0x40 )
  {
    v56 = v63;
    v12 = sub_16A5810((__int64)&v62);
    v11 = v56;
  }
  else
  {
    v12 = 64;
    if ( v62 << (64 - (unsigned __int8)v63) != -1 )
    {
      _BitScanReverse64(&v13, ~(v62 << (64 - (unsigned __int8)v63)));
      v12 = v13 ^ 0x3F;
    }
  }
  v14 = v65;
  if ( v6 <= v12 )
  {
    v18 = 0;
    goto LABEL_36;
  }
  if ( v65 > 0x40 )
  {
    v54 = v65;
    v31 = sub_16A57B0((__int64)&v64);
    v14 = v54;
    if ( v6 > v31 )
      goto LABEL_27;
    v32 = *(unsigned __int8 *)(v4 + 16);
    v18 = 1;
    if ( (unsigned __int8)v32 <= 0x17u )
      goto LABEL_28;
    v19 = v32 - 24;
    if ( v32 == 52 )
    {
LABEL_27:
      v18 = 2;
      goto LABEL_28;
    }
  }
  else
  {
    v15 = v65;
    if ( v64 )
    {
      _BitScanReverse64(&v16, v64);
      v15 = v65 - 64 + (v16 ^ 0x3F);
    }
    if ( v6 > v15 )
    {
      v18 = 2;
      goto LABEL_31;
    }
    v17 = *(unsigned __int8 *)(v4 + 16);
    v18 = 1;
    if ( (unsigned __int8)v17 <= 0x17u )
      goto LABEL_38;
    v19 = v17 - 24;
    if ( v17 == 52 )
    {
      v18 = 2;
      v11 = v10;
      goto LABEL_31;
    }
  }
  if ( v19 == 53 )
  {
    v18 = 1;
    if ( (unsigned int)(*(_DWORD *)(a3 + 28) - *(_DWORD *)(a3 + 32)) > 0xF )
      goto LABEL_36;
    v34 = *(__int64 **)(a3 + 16);
    v35 = *(__int64 **)(a3 + 8);
    if ( v34 == v35 )
    {
      v36 = &v34[*(unsigned int *)(a3 + 28)];
      if ( v34 == v36 )
      {
        v39 = *(__int64 **)(a3 + 16);
        v37 = v39;
      }
      else
      {
        v37 = *(__int64 **)(a3 + 16);
        do
        {
          if ( v4 == *v37 )
            break;
          ++v37;
        }
        while ( v36 != v37 );
        v39 = &v34[*(unsigned int *)(a3 + 28)];
      }
    }
    else
    {
      v36 = &v34[*(unsigned int *)(a3 + 24)];
      v37 = sub_16CC9F0(a3, v4);
      if ( v4 == *v37 )
      {
        v34 = *(__int64 **)(a3 + 16);
        v35 = *(__int64 **)(a3 + 8);
        if ( v34 == v35 )
          v39 = &v34[*(unsigned int *)(a3 + 28)];
        else
          v39 = &v34[*(unsigned int *)(a3 + 24)];
      }
      else
      {
        v34 = *(__int64 **)(a3 + 16);
        v35 = *(__int64 **)(a3 + 8);
        if ( v34 != v35 )
        {
          if ( v36 != &v34[*(unsigned int *)(a3 + 24)] )
          {
LABEL_47:
            v14 = v65;
            v18 = 2;
            goto LABEL_36;
          }
          goto LABEL_100;
        }
        v37 = &v34[*(unsigned int *)(a3 + 28)];
        v39 = v37;
      }
    }
    while ( v39 != v37 && (unsigned __int64)*v37 >= 0xFFFFFFFFFFFFFFFELL )
      ++v37;
    if ( v37 != v36 )
      goto LABEL_47;
    if ( v34 == v35 )
    {
      v40 = &v35[*(unsigned int *)(a3 + 28)];
      v41 = *(_DWORD *)(a3 + 28);
      if ( v40 != v35 )
      {
        v42 = 0;
        while ( v4 != *v35 )
        {
          if ( *v35 == -2 )
            v42 = v35;
          if ( v40 == ++v35 )
          {
            if ( !v42 )
              goto LABEL_112;
            *v42 = v4;
            --*(_DWORD *)(a3 + 32);
            ++*(_QWORD *)a3;
            break;
          }
        }
LABEL_72:
        v43 = 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
        {
          v44 = *(_QWORD **)(v4 - 8);
          v4 = (__int64)&v44[(unsigned __int64)v43 / 8];
        }
        else
        {
          v44 = (_QWORD *)(v4 - v43);
        }
        v45 = (__int64)(0xAAAAAAAAAAAAAAABLL * (v43 >> 3)) >> 2;
        if ( v45 )
        {
          v46 = &v44[12 * v45];
          while ( 1 )
          {
            v47 = *v44;
            if ( (unsigned int)sub_394C0B0(a1, *v44, a3) != 2 && *(_BYTE *)(v47 + 16) != 9 )
              goto LABEL_78;
            v48 = v44[3];
            if ( (unsigned int)sub_394C0B0(a1, v48, a3) != 2 && *(_BYTE *)(v48 + 16) != 9 )
            {
              v44 += 3;
              goto LABEL_78;
            }
            v49 = v44[6];
            if ( (unsigned int)sub_394C0B0(a1, v49, a3) != 2 && *(_BYTE *)(v49 + 16) != 9 )
            {
              v44 += 6;
              goto LABEL_78;
            }
            v50 = v44[9];
            if ( (unsigned int)sub_394C0B0(a1, v50, a3) != 2 && *(_BYTE *)(v50 + 16) != 9 )
            {
              v44 += 9;
              goto LABEL_78;
            }
            v44 += 12;
            if ( v44 == v46 )
              goto LABEL_89;
          }
        }
        v46 = v44;
LABEL_89:
        v51 = v4 - (_QWORD)v46;
        if ( v4 - (_QWORD)v46 != 48 )
        {
          if ( v51 != 72 )
          {
            if ( v51 != 24 )
              goto LABEL_47;
            goto LABEL_92;
          }
          v57 = *v46;
          if ( (unsigned int)sub_394C0B0(a1, *v46, a3) != 2 )
          {
            v44 = v46;
            if ( *(_BYTE *)(v57 + 16) != 9 )
              goto LABEL_78;
          }
          v46 += 3;
        }
        v58 = *v46;
        if ( (unsigned int)sub_394C0B0(a1, *v46, a3) != 2 )
        {
          v44 = v46;
          if ( *(_BYTE *)(v58 + 16) != 9 )
            goto LABEL_78;
        }
        v46 += 3;
LABEL_92:
        v52 = *v46;
        if ( (unsigned int)sub_394C0B0(a1, *v46, a3) == 2 || *(_BYTE *)(v52 + 16) == 9 )
          goto LABEL_47;
        v44 = v46;
LABEL_78:
        v14 = v65;
        v18 = (v44 == (_QWORD *)v4) + 1;
        goto LABEL_36;
      }
LABEL_112:
      if ( v41 < *(_DWORD *)(a3 + 24) )
      {
        *(_DWORD *)(a3 + 28) = v41 + 1;
        *v40 = v4;
        ++*(_QWORD *)a3;
        goto LABEL_72;
      }
    }
LABEL_100:
    sub_16CCBA0(a3, v4);
    goto LABEL_72;
  }
  v18 = 1;
  if ( v19 == 15 )
  {
    v20 = (*(_BYTE *)(v4 + 23) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
    v21 = *(_QWORD *)(v20 + 24);
    v22 = *(_BYTE *)(v21 + 16);
    if ( v22 == 13 )
      goto LABEL_18;
    v18 = 1;
    if ( v22 == 71 )
    {
      v21 = *(_QWORD *)(v21 - 24);
      if ( !v21 )
        BUG();
      if ( *(_BYTE *)(v21 + 16) == 13 )
      {
LABEL_18:
        v23 = *(_DWORD *)(v21 + 32);
        v24 = *(_QWORD *)(v21 + 24);
        v25 = v23 + 1;
        v26 = 1LL << ((unsigned __int8)v23 - 1);
        if ( v23 > 0x40 )
        {
          v38 = v21 + 24;
          if ( (*(_QWORD *)(v24 + 8LL * ((v23 - 1) >> 6)) & v26) == 0 )
          {
            v61 = v14;
            v53 = sub_16A57B0(v38);
            v14 = v61;
            v30 = v25 - v53;
            goto LABEL_23;
          }
          v60 = v14;
          v27 = sub_16A5810(v38);
          v14 = v60;
        }
        else
        {
          if ( (v26 & v24) == 0 )
          {
            v30 = 1;
            if ( v24 )
            {
              _BitScanReverse64(&v24, v24);
              v30 = 65 - (v24 ^ 0x3F);
            }
            goto LABEL_23;
          }
          v27 = 64;
          v28 = ~(v24 << (64 - (unsigned __int8)v23));
          if ( v28 )
          {
            _BitScanReverse64(&v29, v28);
            v27 = v29 ^ 0x3F;
          }
        }
        v30 = v25 - v27;
LABEL_23:
        v18 = (*(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 8 < v30) + 1;
      }
    }
  }
LABEL_36:
  if ( v14 <= 0x40 )
  {
    v10 = v63;
LABEL_38:
    v11 = v10;
    goto LABEL_31;
  }
LABEL_28:
  if ( v64 )
    j_j___libc_free_0_0(v64);
  v11 = v63;
LABEL_31:
  if ( v11 > 0x40 && v62 )
    j_j___libc_free_0_0(v62);
  return v18;
}
