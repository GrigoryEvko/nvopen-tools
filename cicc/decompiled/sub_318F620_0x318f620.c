// Function: sub_318F620
// Address: 0x318f620
//
__int64 __fastcall sub_318F620(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned int v6; // ebx
  __int64 v7; // rax
  unsigned __int64 v8; // r10
  __int64 v9; // r9
  unsigned int v10; // ebx
  unsigned int v11; // esi
  unsigned int v12; // eax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // r15d
  __int64 v18; // r8
  unsigned int v19; // edi
  unsigned __int64 v20; // rax
  int v21; // eax
  int v22; // esi
  unsigned __int8 *v23; // r12
  __int64 v24; // rdi
  unsigned int v25; // eax
  unsigned int v26; // ebx
  __int64 v27; // r8
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rcx
  unsigned int v31; // eax
  unsigned int v32; // eax
  int v33; // eax
  unsigned int v34; // esi
  unsigned __int8 **v35; // rax
  __int64 v36; // rdi
  int v37; // eax
  char v38; // dl
  __int64 v39; // r15
  unsigned __int8 *v40; // rbx
  __int64 v41; // r15
  unsigned __int8 *v42; // r15
  unsigned __int8 *v43; // r12
  unsigned __int8 *v44; // r12
  unsigned __int8 *v45; // r12
  unsigned __int8 *v46; // r12
  signed __int64 v47; // rax
  unsigned __int8 *v48; // rbx
  unsigned __int8 *v49; // rbx
  unsigned __int8 *v50; // rbx
  unsigned int v51; // [rsp+Ch] [rbp-64h]
  unsigned int v52; // [rsp+10h] [rbp-60h]
  unsigned int v53; // [rsp+10h] [rbp-60h]
  unsigned __int64 v54; // [rsp+10h] [rbp-60h]
  unsigned int v55; // [rsp+18h] [rbp-58h]
  int v56; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v57; // [rsp+18h] [rbp-58h]
  unsigned __int64 v58; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v59; // [rsp+28h] [rbp-48h]
  unsigned __int64 v60; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v61; // [rsp+38h] [rbp-38h]

  v6 = *(_DWORD *)(*((_QWORD *)a2 + 1) + 8LL) >> 8;
  v55 = v6 - (*(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 8);
  v7 = sub_B43CC0(*(_QWORD *)(a1 + 8));
  v59 = v6;
  v8 = v7;
  if ( v6 > 0x40 )
  {
    v54 = v7;
    sub_C43690((__int64)&v58, 0, 0);
    v61 = v6;
    sub_C43690((__int64)&v60, 0, 0);
    v8 = v54;
  }
  else
  {
    v58 = 0;
    v61 = v6;
    v60 = 0;
  }
  sub_9AC1B0((__int64)a2, &v58, v8, 0, 0, 0, 0, 1);
  v10 = v59;
  v11 = v59;
  if ( v59 > 0x40 )
  {
    v52 = v59;
    v12 = sub_C44500((__int64)&v58);
    v11 = v52;
    goto LABEL_16;
  }
  if ( !v59 )
  {
    v12 = 0;
LABEL_16:
    v14 = v61;
    v15 = v61;
    if ( v55 <= v12 )
      goto LABEL_7;
    goto LABEL_17;
  }
  v12 = 64;
  if ( v58 << (64 - (unsigned __int8)v59) == -1 )
    goto LABEL_16;
  _BitScanReverse64(&v13, ~(v58 << (64 - (unsigned __int8)v59)));
  v14 = v61;
  v15 = v61;
  if ( v55 <= ((unsigned int)v13 ^ 0x3F) )
  {
LABEL_7:
    v16 = 0;
    goto LABEL_8;
  }
LABEL_17:
  if ( (unsigned int)v14 > 0x40 )
  {
    v51 = v14;
    v53 = v14;
    v32 = sub_C444A0((__int64)&v60);
    v14 = v53;
    v15 = v51;
    if ( v55 > v32 )
      goto LABEL_37;
    v33 = *a2;
    if ( (unsigned __int8)v33 <= 0x1Cu )
    {
      v16 = 1;
      goto LABEL_38;
    }
    v22 = v33 - 29;
    if ( v33 == 59 )
    {
LABEL_37:
      v16 = 2;
      goto LABEL_38;
    }
  }
  else
  {
    v18 = (unsigned int)(v14 - 64);
    v19 = v14;
    if ( v60 )
    {
      _BitScanReverse64(&v20, v60);
      v19 = v18 + (v20 ^ 0x3F);
    }
    if ( v55 > v19 )
    {
      v16 = 2;
      goto LABEL_11;
    }
    v21 = *a2;
    if ( (unsigned __int8)v21 <= 0x1Cu )
    {
      v16 = 1;
      goto LABEL_10;
    }
    v22 = v21 - 29;
    if ( v21 == 59 )
    {
      v16 = 2;
      goto LABEL_10;
    }
  }
  if ( v22 == 55 )
  {
    v34 = *(_DWORD *)(a3 + 20);
    v16 = 1;
    if ( v34 - *(_DWORD *)(a3 + 24) > 0xF )
      goto LABEL_8;
    if ( *(_BYTE *)(a3 + 28) )
    {
      v35 = *(unsigned __int8 ***)(a3 + 8);
      v14 = (__int64)&v35[v34];
      if ( v35 != (unsigned __int8 **)v14 )
      {
        while ( a2 != *v35 )
        {
          if ( (unsigned __int8 **)v14 == ++v35 )
            goto LABEL_83;
        }
        goto LABEL_52;
      }
LABEL_83:
      if ( v34 < *(_DWORD *)(a3 + 16) )
      {
        *(_DWORD *)(a3 + 20) = v34 + 1;
        *(_QWORD *)v14 = a2;
        ++*(_QWORD *)a3;
LABEL_63:
        v39 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
        if ( (a2[7] & 0x40) != 0 )
        {
          v40 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          v57 = &v40[v39];
        }
        else
        {
          v57 = a2;
          v40 = &a2[-v39];
        }
        v41 = v39 >> 7;
        if ( v41 )
        {
          v42 = &v40[128 * v41];
          while ( 1 )
          {
            v46 = *(unsigned __int8 **)v40;
            if ( (unsigned int)sub_318F620(a1, *(_QWORD *)v40, a3) != 2 && (unsigned int)*v46 - 12 > 1 )
              goto LABEL_76;
            v43 = (unsigned __int8 *)*((_QWORD *)v40 + 4);
            if ( (unsigned int)sub_318F620(a1, v43, a3) != 2 && (unsigned int)*v43 - 12 > 1 )
            {
              v40 += 32;
              goto LABEL_76;
            }
            v44 = (unsigned __int8 *)*((_QWORD *)v40 + 8);
            if ( (unsigned int)sub_318F620(a1, v44, a3) != 2 && (unsigned int)*v44 - 12 > 1 )
            {
              v40 += 64;
              goto LABEL_76;
            }
            v45 = (unsigned __int8 *)*((_QWORD *)v40 + 12);
            if ( (unsigned int)sub_318F620(a1, v45, a3) != 2 && (unsigned int)*v45 - 12 > 1 )
            {
              v40 += 96;
              goto LABEL_76;
            }
            v40 += 128;
            if ( v40 == v42 )
              goto LABEL_88;
          }
        }
        v42 = v40;
LABEL_88:
        v47 = v57 - v42;
        if ( v57 - v42 != 64 )
        {
          if ( v47 != 96 )
          {
            if ( v47 != 32 )
            {
LABEL_91:
              LODWORD(v14) = v61;
              v16 = 2;
              goto LABEL_8;
            }
            goto LABEL_99;
          }
          v48 = *(unsigned __int8 **)v42;
          if ( (unsigned int)sub_318F620(a1, *(_QWORD *)v42, a3) != 2 && (unsigned int)*v48 - 12 > 1 )
            goto LABEL_101;
          v42 += 32;
        }
        v49 = *(unsigned __int8 **)v42;
        if ( (unsigned int)sub_318F620(a1, *(_QWORD *)v42, a3) != 2 && (unsigned int)*v49 - 12 > 1 )
          goto LABEL_101;
        v42 += 32;
LABEL_99:
        v50 = *(unsigned __int8 **)v42;
        if ( (unsigned int)sub_318F620(a1, *(_QWORD *)v42, a3) == 2 || (unsigned int)*v50 - 12 <= 1 )
          goto LABEL_91;
LABEL_101:
        v40 = v42;
LABEL_76:
        LODWORD(v14) = v61;
        v16 = (v57 == v40) + 1;
        goto LABEL_8;
      }
    }
    sub_C8CC70(a3, (__int64)a2, v14, v15, v18, v9);
    if ( v38 )
      goto LABEL_63;
    LODWORD(v15) = v61;
LABEL_52:
    LODWORD(v14) = v15;
    v16 = 2;
    goto LABEL_8;
  }
  v16 = 1;
  if ( v22 == 17 )
  {
    v23 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v24 = *((_QWORD *)v23 + 4);
    if ( *(_BYTE *)v24 == 17 )
      goto LABEL_28;
    v16 = 1;
    if ( *(_BYTE *)v24 == 78 )
    {
      v24 = *(_QWORD *)(v24 - 32);
      if ( !v24 )
        BUG();
      if ( *(_BYTE *)v24 == 17 )
      {
LABEL_28:
        v25 = *(_DWORD *)(v24 + 32);
        v26 = v25 + 1;
        v27 = 1LL << ((unsigned __int8)v25 - 1);
        v28 = *(_QWORD *)(v24 + 24);
        if ( v25 > 0x40 )
        {
          v36 = v24 + 24;
          v56 = v14;
          if ( (*(_QWORD *)(v28 + 8LL * ((v25 - 1) >> 6)) & v27) != 0 )
            v37 = sub_C44500(v36);
          else
            v37 = sub_C444A0(v36);
          LODWORD(v14) = v56;
          v31 = v26 - v37;
        }
        else if ( (v27 & v28) != 0 )
        {
          if ( v25 )
          {
            v29 = ~(v28 << (64 - (unsigned __int8)v25));
            if ( v29 )
            {
              _BitScanReverse64(&v30, v29);
              v31 = v26 - (v30 ^ 0x3F);
            }
            else
            {
              v31 = v25 - 63;
            }
          }
          else
          {
            v31 = 1;
          }
        }
        else
        {
          v31 = 1;
          if ( v28 )
          {
            _BitScanReverse64(&v28, v28);
            v31 = 65 - (v28 ^ 0x3F);
          }
        }
        v16 = (*(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 8 < v31) + 1;
      }
    }
  }
LABEL_8:
  if ( (unsigned int)v14 <= 0x40 )
  {
    v10 = v59;
LABEL_10:
    v11 = v10;
    goto LABEL_11;
  }
LABEL_38:
  if ( v60 )
    j_j___libc_free_0_0(v60);
  v11 = v59;
LABEL_11:
  if ( v11 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  return v16;
}
