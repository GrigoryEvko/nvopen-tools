// Function: sub_3026A90
// Address: 0x3026a90
//
void __fastcall sub_3026A90(__int64 a1, unsigned __int8 *a2, int a3, __int64 a4)
{
  __int64 v7; // r13
  __int64 v8; // rsi
  char v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r9
  unsigned int v14; // eax
  int v15; // edx
  bool v16; // al
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rax
  int v21; // r10d
  unsigned __int8 v22; // al
  __int64 *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned int v28; // edx
  unsigned int v29; // eax
  int v30; // ecx
  __int64 v31; // rax
  int v32; // r13d
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  unsigned int v37; // eax
  int v38; // edx
  unsigned int v39; // eax
  int v40; // edx
  _BYTE *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  unsigned __int8 *v45; // rax
  int v46; // r14d
  unsigned __int8 *v47; // r13
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  unsigned __int8 *v51; // rax
  __int64 v52; // r8
  int v53; // r14d
  unsigned __int8 *v54; // r13
  __int64 v55; // r9
  int v56; // r10d
  __int64 v57; // r12
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rax
  unsigned __int64 v62; // rcx
  unsigned int v63; // eax
  int v64; // edx
  __int64 v65; // [rsp+8h] [rbp-68h]
  int v66; // [rsp+10h] [rbp-60h]
  int v67; // [rsp+10h] [rbp-60h]
  int v68; // [rsp+10h] [rbp-60h]
  int v69; // [rsp+10h] [rbp-60h]
  int v70; // [rsp+10h] [rbp-60h]
  __int64 v71; // [rsp+18h] [rbp-58h]
  int v72; // [rsp+18h] [rbp-58h]
  int v73; // [rsp+18h] [rbp-58h]
  __int64 v74; // [rsp+18h] [rbp-58h]
  __int64 v75; // [rsp+18h] [rbp-58h]
  int v76; // [rsp+18h] [rbp-58h]
  __int64 v77; // [rsp+18h] [rbp-58h]
  __int64 v78; // [rsp+20h] [rbp-50h] BYREF
  int v79; // [rsp+28h] [rbp-48h]
  unsigned __int64 v80; // [rsp+30h] [rbp-40h] BYREF
  __int64 v81; // [rsp+38h] [rbp-38h]

  v7 = sub_31DA930();
  v71 = *((_QWORD *)a2 + 1);
  v8 = v71;
  v9 = sub_AE5020(v7, v71);
  v10 = sub_9208B0(v7, v71);
  v81 = v11;
  v80 = ((1LL << v9) + ((unsigned __int64)(v10 + 7) >> 3) - 1) >> v9 << v9;
  v12 = sub_CA1930(&v80);
  v72 = v12;
  LODWORD(v13) = v12;
  if ( (unsigned int)*a2 - 12 <= 1 || (v65 = v12, v16 = sub_AC30F0((__int64)a2), v13 = v65, v16) )
  {
    if ( !a3 )
      a3 = v13;
    if ( a3 > 0 )
    {
      v14 = *(_DWORD *)(a4 + 160);
      v15 = 0;
      do
      {
        ++v15;
        *(_BYTE *)(*(_QWORD *)(a4 + 8) + v14) = 0;
        v14 = *(_DWORD *)(a4 + 160) + 1;
        *(_DWORD *)(a4 + 160) = v14;
      }
      while ( a3 != v15 );
    }
    return;
  }
  v20 = *((_QWORD *)a2 + 1);
  v21 = v72;
  v79 = a3;
  v78 = a4;
  v22 = *(_BYTE *)(v20 + 8);
  if ( v22 == 14 )
  {
    if ( *a2 > 3u )
    {
      if ( *a2 != 5 )
      {
LABEL_37:
        if ( (int)v13 > 0 )
        {
          v37 = *(_DWORD *)(a4 + 160);
          v38 = 0;
          do
          {
            ++v38;
            *(_BYTE *)(*(_QWORD *)(a4 + 8) + v37) = 0;
            v37 = *(_DWORD *)(a4 + 160) + 1;
            *(_DWORD *)(a4 + 160) = v37;
          }
          while ( v21 != v38 );
        }
        return;
      }
      v45 = sub_BD3990(a2, v8);
      v46 = *(_DWORD *)(a4 + 160);
      v47 = v45;
      v48 = *(unsigned int *)(a4 + 40);
      v13 = v65;
      v21 = v72;
      if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
      {
        sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v48 + 1, 4u, v19, v65);
        v48 = *(unsigned int *)(a4 + 40);
        v21 = v72;
        v13 = v65;
      }
      *(_DWORD *)(*(_QWORD *)(a4 + 32) + 4 * v48) = v46;
      v49 = *(unsigned int *)(a4 + 72);
      v50 = *(unsigned int *)(a4 + 76);
      ++*(_DWORD *)(a4 + 40);
      if ( v49 + 1 > v50 )
      {
        v68 = v21;
        v75 = v13;
        sub_C8D5F0(a4 + 64, (const void *)(a4 + 80), v49 + 1, 8u, v19, v13);
        v49 = *(unsigned int *)(a4 + 72);
        v21 = v68;
        v13 = v75;
      }
      *(_QWORD *)(*(_QWORD *)(a4 + 64) + 8 * v49) = v47;
    }
    else
    {
      v31 = *(unsigned int *)(a4 + 40);
      v32 = *(_DWORD *)(a4 + 160);
      if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
      {
        sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v31 + 1, 4u, v19, v65);
        v31 = *(unsigned int *)(a4 + 40);
        v21 = v72;
        v13 = v65;
      }
      *(_DWORD *)(*(_QWORD *)(a4 + 32) + 4 * v31) = v32;
      v33 = *(unsigned int *)(a4 + 72);
      v34 = *(unsigned int *)(a4 + 76);
      ++*(_DWORD *)(a4 + 40);
      if ( v33 + 1 > v34 )
      {
        v67 = v21;
        v74 = v13;
        sub_C8D5F0(a4 + 64, (const void *)(a4 + 80), v33 + 1, 8u, v19, v13);
        v33 = *(unsigned int *)(a4 + 72);
        v21 = v67;
        v13 = v74;
      }
      *(_QWORD *)(*(_QWORD *)(a4 + 64) + 8 * v33) = a2;
    }
    v35 = *(unsigned int *)(a4 + 120);
    v36 = *(unsigned int *)(a4 + 124);
    ++*(_DWORD *)(a4 + 72);
    if ( v35 + 1 > v36 )
    {
      v66 = v21;
      v73 = v13;
      sub_C8D5F0(a4 + 112, (const void *)(a4 + 128), v35 + 1, 8u, v19, v13);
      v35 = *(unsigned int *)(a4 + 120);
      v21 = v66;
      LODWORD(v13) = v73;
    }
    *(_QWORD *)(*(_QWORD *)(a4 + 112) + 8 * v35) = a2;
    ++*(_DWORD *)(a4 + 120);
    goto LABEL_37;
  }
  if ( v22 <= 0xEu )
  {
    if ( v22 <= 3u )
    {
      v23 = (__int64 *)(a2 + 24);
      if ( *((void **)a2 + 3) == sub_C33340() )
        sub_C3E660((__int64)&v80, (__int64)v23);
      else
        sub_C3A850((__int64)&v80, v23);
      sub_3020200(&v78, (__int64)&v80, v24, v25, v26, v27);
      if ( (unsigned int)v81 > 0x40 && v80 )
        j_j___libc_free_0_0(v80);
      return;
    }
    if ( v22 == 12 )
    {
      if ( *a2 == 17 )
      {
        sub_3020200(&v78, (__int64)(a2 + 24), v17, v18, v19, v65);
        return;
      }
      if ( *a2 == 5 )
      {
        v41 = (_BYTE *)sub_97B670(a2, v7, 0);
        if ( *v41 == 17 )
        {
          sub_3020200(&v78, (__int64)(v41 + 24), v42, v43, v44, v65);
          return;
        }
        if ( *((_WORD *)a2 + 1) == 47 )
        {
          v51 = sub_BD3990(*(unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)], v7);
          v53 = *(_DWORD *)(a4 + 160);
          v54 = v51;
          v55 = v65;
          v56 = v72;
          v57 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v58 = *(unsigned int *)(a4 + 40);
          if ( v58 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
          {
            sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v58 + 1, 4u, v52, v65);
            v58 = *(unsigned int *)(a4 + 40);
            v56 = v72;
            v55 = v65;
          }
          *(_DWORD *)(*(_QWORD *)(a4 + 32) + 4 * v58) = v53;
          v59 = *(unsigned int *)(a4 + 72);
          v60 = *(unsigned int *)(a4 + 76);
          ++*(_DWORD *)(a4 + 40);
          if ( v59 + 1 > v60 )
          {
            v70 = v56;
            v77 = v55;
            sub_C8D5F0(a4 + 64, (const void *)(a4 + 80), v59 + 1, 8u, v52, v55);
            v59 = *(unsigned int *)(a4 + 72);
            v56 = v70;
            v55 = v77;
          }
          *(_QWORD *)(*(_QWORD *)(a4 + 64) + 8 * v59) = v54;
          v61 = *(unsigned int *)(a4 + 120);
          v62 = *(unsigned int *)(a4 + 124);
          ++*(_DWORD *)(a4 + 72);
          if ( v61 + 1 > v62 )
          {
            v69 = v56;
            v76 = v55;
            sub_C8D5F0(a4 + 112, (const void *)(a4 + 128), v61 + 1, 8u, v52, v55);
            v61 = *(unsigned int *)(a4 + 120);
            v56 = v69;
            LODWORD(v55) = v76;
          }
          *(_QWORD *)(*(_QWORD *)(a4 + 112) + 8 * v61) = v57;
          ++*(_DWORD *)(a4 + 120);
          if ( (int)v55 > 0 )
          {
            v63 = *(_DWORD *)(a4 + 160);
            v64 = 0;
            do
            {
              ++v64;
              *(_BYTE *)(*(_QWORD *)(a4 + 8) + v63) = 0;
              v63 = *(_DWORD *)(a4 + 160) + 1;
              *(_DWORD *)(a4 + 160) = v63;
            }
            while ( v56 != v64 );
          }
          return;
        }
      }
LABEL_71:
      BUG();
    }
LABEL_72:
    BUG();
  }
  if ( (unsigned __int8)(v22 - 15) > 2u )
    goto LABEL_72;
  v28 = *a2;
  if ( (unsigned __int8)v28 > 8u && (v28 == 15 || v28 <= 0xB || v28 == 16) )
  {
    sub_3026630(a1, (__int64)a2, a4);
    if ( (int)v65 < a3 )
    {
      v29 = *(_DWORD *)(a4 + 160);
      v30 = 0;
      do
      {
        ++v30;
        *(_BYTE *)(*(_QWORD *)(a4 + 8) + v29) = 0;
        v29 = *(_DWORD *)(a4 + 160) + 1;
        *(_DWORD *)(a4 + 160) = v29;
      }
      while ( a3 - (_DWORD)v65 != v30 );
    }
  }
  else
  {
    if ( *a2 != 14 )
      goto LABEL_71;
    if ( a3 > 0 )
    {
      v39 = *(_DWORD *)(a4 + 160);
      v40 = 0;
      do
      {
        ++v40;
        *(_BYTE *)(*(_QWORD *)(a4 + 8) + v39) = 0;
        v39 = *(_DWORD *)(a4 + 160) + 1;
        *(_DWORD *)(a4 + 160) = v39;
      }
      while ( a3 != v40 );
    }
  }
}
