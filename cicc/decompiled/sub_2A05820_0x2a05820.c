// Function: sub_2A05820
// Address: 0x2a05820
//
__int64 __fastcall sub_2A05820(int *a1, unsigned __int8 *a2)
{
  char v4; // cl
  int *v5; // rdi
  int v6; // esi
  unsigned int v7; // edx
  unsigned __int8 **v8; // rax
  unsigned __int8 *v9; // r9
  unsigned int v11; // esi
  unsigned int v12; // eax
  unsigned __int8 **v13; // r8
  int v14; // edx
  unsigned int v15; // edi
  __int64 v16; // rcx
  __int64 v17; // rdx
  int v18; // eax
  unsigned __int8 *v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // r14d
  unsigned __int8 *v22; // rdx
  __int64 v23; // rax
  _BYTE *v24; // rax
  _BYTE *v25; // rax
  bool v26; // zf
  int v27; // r10d
  unsigned __int8 *v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // r8
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // eax
  char v39; // dl
  _QWORD *v40; // r8
  __int64 v41; // rax
  int v42; // edx
  unsigned int v43; // eax
  unsigned __int8 *v44; // rsi
  int v45; // edx
  unsigned int v46; // eax
  unsigned __int8 *v47; // rsi
  int v48; // r9d
  unsigned __int8 **v49; // rdi
  int v50; // edx
  int v51; // edx
  int v52; // r9d
  unsigned __int8 *v53; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int8 *v55; // [rsp+18h] [rbp-38h] BYREF
  __int64 v56; // [rsp+20h] [rbp-30h]
  __int64 v57[5]; // [rsp+28h] [rbp-28h] BYREF

  v4 = a1[8] & 1;
  if ( v4 )
  {
    v5 = a1 + 10;
    v6 = 3;
  }
  else
  {
    v11 = a1[12];
    v5 = (int *)*((_QWORD *)a1 + 5);
    if ( !v11 )
    {
      v12 = a1[8];
      ++*((_QWORD *)a1 + 3);
      v13 = 0;
      v14 = (v12 >> 1) + 1;
LABEL_9:
      v15 = 3 * v11;
      goto LABEL_10;
    }
    v6 = v11 - 1;
  }
  v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (unsigned __int8 **)&v5[4 * v7];
  v9 = *v8;
  if ( a2 == *v8 )
    return (__int64)v8[1];
  v27 = 1;
  v13 = 0;
  while ( v9 != (unsigned __int8 *)-4096LL )
  {
    if ( v9 == (unsigned __int8 *)-8192LL && !v13 )
      v13 = v8;
    v7 = v6 & (v27 + v7);
    v8 = (unsigned __int8 **)&v5[4 * v7];
    v9 = *v8;
    if ( a2 == *v8 )
      return (__int64)v8[1];
    ++v27;
  }
  if ( !v13 )
    v13 = v8;
  v12 = a1[8];
  ++*((_QWORD *)a1 + 3);
  v14 = (v12 >> 1) + 1;
  if ( !v4 )
  {
    v11 = a1[12];
    goto LABEL_9;
  }
  v15 = 12;
  v11 = 4;
LABEL_10:
  if ( 4 * v14 < v15 )
  {
    v16 = v11 - a1[9] - v14;
    if ( (unsigned int)v16 > v11 >> 3 )
      goto LABEL_12;
    sub_2A05110((__int64)(a1 + 6), v11);
    if ( (a1[8] & 1) != 0 )
    {
      v16 = (__int64)(a1 + 10);
      v45 = 3;
      goto LABEL_62;
    }
    v51 = a1[12];
    v16 = *((_QWORD *)a1 + 5);
    if ( v51 )
    {
      v45 = v51 - 1;
LABEL_62:
      v46 = v45 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (unsigned __int8 **)(v16 + 16LL * v46);
      v47 = *v13;
      if ( a2 != *v13 )
      {
        v48 = 1;
        v49 = 0;
        while ( v47 != (unsigned __int8 *)-4096LL )
        {
          if ( v47 == (unsigned __int8 *)-8192LL && !v49 )
            v49 = v13;
          v46 = v45 & (v48 + v46);
          v13 = (unsigned __int8 **)(v16 + 16LL * v46);
          v47 = *v13;
          if ( a2 == *v13 )
            goto LABEL_59;
          ++v48;
        }
LABEL_65:
        if ( v49 )
          v13 = v49;
        goto LABEL_59;
      }
      goto LABEL_59;
    }
LABEL_97:
    a1[8] = (2 * ((unsigned int)a1[8] >> 1) + 2) | a1[8] & 1;
    BUG();
  }
  sub_2A05110((__int64)(a1 + 6), 2 * v11);
  if ( (a1[8] & 1) != 0 )
  {
    v16 = (__int64)(a1 + 10);
    v42 = 3;
  }
  else
  {
    v50 = a1[12];
    v16 = *((_QWORD *)a1 + 5);
    if ( !v50 )
      goto LABEL_97;
    v42 = v50 - 1;
  }
  v43 = v42 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (unsigned __int8 **)(v16 + 16LL * v43);
  v44 = *v13;
  if ( a2 != *v13 )
  {
    v52 = 1;
    v49 = 0;
    while ( v44 != (unsigned __int8 *)-4096LL )
    {
      if ( !v49 && v44 == (unsigned __int8 *)-8192LL )
        v49 = v13;
      v43 = v42 & (v52 + v43);
      v13 = (unsigned __int8 **)(v16 + 16LL * v43);
      v44 = *v13;
      if ( a2 == *v13 )
        goto LABEL_59;
      ++v52;
    }
    goto LABEL_65;
  }
LABEL_59:
  v12 = a1[8];
LABEL_12:
  v17 = 2 * (v12 >> 1) + 2;
  a1[8] = v17 | v12 & 1;
  if ( *v13 != (unsigned __int8 *)-4096LL )
    --a1[9];
  *v13 = a2;
  v13[1] = *(unsigned __int8 **)a1;
  if ( (unsigned __int8)sub_D48480(*((_QWORD *)a1 + 1), (__int64)a2, v17, v16) )
  {
    v57[0] = (__int64)a2;
    v25 = sub_2A05530((__int64)(a1 + 6), v57);
    v26 = v25[4] == 0;
    *(_DWORD *)v25 = 0;
    if ( v26 )
      v25[4] = 1;
    return *(_QWORD *)v25;
  }
  else
  {
    v18 = *a2;
    if ( (unsigned __int8)v18 <= 0x1Cu )
      return *(_QWORD *)a1;
    if ( (_BYTE)v18 == 84 )
    {
      v32 = *((_QWORD *)a1 + 1);
      if ( **(_QWORD **)(v32 + 32) != *((_QWORD *)a2 + 5) )
        return *(_QWORD *)a1;
      v33 = sub_D47930(v32);
      v34 = *((_QWORD *)a2 - 1);
      v35 = v33;
      if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) != 0 )
      {
        v36 = 0;
        while ( v35 != *(_QWORD *)(v34 + 32LL * *((unsigned int *)a2 + 18) + 8 * v36) )
        {
          if ( (*((_DWORD *)a2 + 1) & 0x7FFFFFF) == (_DWORD)++v36 )
            goto LABEL_80;
        }
        v37 = 32 * v36;
      }
      else
      {
LABEL_80:
        v37 = 0x1FFFFFFFE0LL;
      }
      v56 = sub_2A05820(a1, *(_QWORD *)(v34 + v37));
      if ( BYTE4(v56) == *((_BYTE *)a1 + 4) && ((v38 = *a1, !BYTE4(v56)) || (_DWORD)v56 == v38) )
      {
        v57[0] = *(_QWORD *)a1;
        v39 = *((_BYTE *)a1 + 4);
      }
      else
      {
        v38 = v56 + 1;
        v39 = 1;
        if ( (int)v56 + 1 > (unsigned int)a1[4] )
        {
          v39 = *((_BYTE *)a1 + 4);
          v57[0] = *(_QWORD *)a1;
          v38 = *a1;
        }
      }
      LODWORD(v57[0]) = v38;
      BYTE4(v57[0]) = v39;
      v53 = a2;
      v55 = (unsigned __int8 *)v57[0];
      v40 = sub_2A05530((__int64)(a1 + 6), (__int64 *)&v53);
      v41 = (__int64)v55;
      *v40 = v55;
      return v41;
    }
    else
    {
      if ( (unsigned __int8)(v18 - 82) > 1u && (unsigned int)(v18 - 42) > 0x11 )
      {
        if ( (unsigned int)(v18 - 67) <= 0xC )
        {
          if ( (a2[7] & 0x40) != 0 )
            v28 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v28 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v29 = sub_2A05820(a1, *(_QWORD *)v28);
          v57[0] = (__int64)a2;
          v56 = v29;
          v30 = sub_2A05530((__int64)(a1 + 6), v57);
          v31 = v56;
          *v30 = v56;
          return v31;
        }
        return *(_QWORD *)a1;
      }
      if ( (a2[7] & 0x40) != 0 )
        v19 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v19 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v20 = sub_2A05820(a1, *(_QWORD *)v19);
      v56 = v20;
      v21 = v20;
      if ( BYTE4(v20) == *((_BYTE *)a1 + 4) && (!BYTE4(v20) || *a1 == (_DWORD)v20) )
        return *(_QWORD *)a1;
      v22 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v23 = sub_2A05820(a1, *((_QWORD *)v22 + 4));
      v57[0] = v23;
      if ( BYTE4(v23) == *((_BYTE *)a1 + 4) && (!BYTE4(v23) || *a1 == (_DWORD)v23) )
        return *(_QWORD *)a1;
      v55 = a2;
      if ( (unsigned int)v23 >= v21 )
        v21 = v23;
      v24 = sub_2A05530((__int64)(a1 + 6), (__int64 *)&v55);
      v24[4] = 1;
      *(_DWORD *)v24 = v21;
      return *(_QWORD *)v24;
    }
  }
}
