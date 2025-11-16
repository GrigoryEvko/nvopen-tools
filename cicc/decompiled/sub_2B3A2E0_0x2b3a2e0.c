// Function: sub_2B3A2E0
// Address: 0x2b3a2e0
//
void __fastcall sub_2B3A2E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  unsigned int v9; // esi
  unsigned int v10; // r14d
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r11
  unsigned __int64 v13; // r10
  const void *v14; // rax
  unsigned int *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned int *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  int *v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  int *v28; // rax
  int *v29; // rsi
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  int *v32; // rax
  int *v33; // rdx
  int *v34; // rsi
  int *v35; // rdx
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned int *v41; // rax
  __int64 v42; // rdx
  unsigned __int64 v43; // rdx
  unsigned int *v44; // rax
  unsigned int *i; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r9
  __int64 v49; // rax
  int *v50; // rdx
  int *v51; // rcx
  int v52; // esi
  int v53; // edi
  int *v54; // rsi
  unsigned __int64 v55; // [rsp+0h] [rbp-90h]
  unsigned __int64 v56; // [rsp+0h] [rbp-90h]
  __int64 v57; // [rsp+8h] [rbp-88h]
  int v58; // [rsp+8h] [rbp-88h]
  int v59; // [rsp+10h] [rbp-80h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  int v61; // [rsp+18h] [rbp-78h]
  unsigned __int64 v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+18h] [rbp-78h]
  int v64; // [rsp+18h] [rbp-78h]
  __int64 v65; // [rsp+18h] [rbp-78h]
  unsigned __int64 v66; // [rsp+18h] [rbp-78h]
  int *v67; // [rsp+20h] [rbp-70h] BYREF
  __int64 v68; // [rsp+28h] [rbp-68h]
  _BYTE v69[96]; // [rsp+30h] [rbp-60h] BYREF

  v6 = a3;
  v9 = *(_DWORD *)(a1 + 8);
  if ( (_BYTE)a4 )
  {
    v10 = a3;
    v11 = (unsigned int)a3;
    v67 = (int *)v69;
    v68 = 0xC00000000LL;
    if ( v9 )
    {
      v12 = v9;
      v13 = 0;
      a6 = 0;
      if ( v9 <= 0xC )
      {
LABEL_4:
        v61 = v12;
        v14 = (const void *)(*(_QWORD *)a1 + 4 * a6);
        if ( v14 != (const void *)(4 * v12 + *(_QWORD *)a1) )
        {
          v59 = v13;
          v55 = v11;
          v57 = a6;
          memcpy(&v67[v13], v14, 4 * v12 - 4 * a6);
          v11 = v55;
          a6 = v57;
          v9 = v61 + v68 - v59;
        }
        LODWORD(v68) = v9;
        *(_DWORD *)(a1 + 8) = a6;
        goto LABEL_7;
      }
      v63 = (unsigned int)a3;
      sub_C8D5F0((__int64)&v67, v69, v9, 4u, (unsigned int)a3, 0);
      v13 = (unsigned int)v68;
      v11 = v63;
      if ( (unsigned int)v68 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v12 = *(unsigned int *)(a1 + 8);
        a6 = (unsigned int)v68;
        v9 = *(_DWORD *)(a1 + 8);
        if ( v12 <= (unsigned int)v68 )
          a6 = *(unsigned int *)(a1 + 8);
      }
      else
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), (unsigned int)v68, 4u, v63, v48);
        v12 = *(unsigned int *)(a1 + 8);
        v13 = (unsigned int)v68;
        v11 = v63;
        a6 = (unsigned int)v68;
        v9 = *(_DWORD *)(a1 + 8);
        if ( v12 <= (unsigned int)v68 )
          a6 = *(unsigned int *)(a1 + 8);
      }
      if ( a6 )
      {
        v49 = 0;
        do
        {
          v50 = (int *)(v49 * 4 + *(_QWORD *)a1);
          v51 = &v67[v49++];
          v52 = *v51;
          *v51 = *v50;
          *v50 = v52;
        }
        while ( a6 != v49 );
        v12 = *(unsigned int *)(a1 + 8);
        v13 = (unsigned int)v68;
        v9 = *(_DWORD *)(a1 + 8);
      }
      if ( v13 <= v12 )
      {
        if ( v13 >= v12 )
          goto LABEL_7;
        goto LABEL_4;
      }
      v53 = v12;
      v54 = &v67[a6];
      if ( v54 != &v67[v13] )
      {
        v60 = a6;
        v56 = v11;
        v58 = v13;
        v64 = v12;
        memcpy((void *)(*(_QWORD *)a1 + 4 * v12), v54, 4 * v13 - 4 * a6);
        v53 = *(_DWORD *)(a1 + 8);
        v11 = v56;
        LODWORD(v13) = v58;
        a6 = v60;
        LODWORD(v12) = v64;
      }
      LODWORD(v68) = a6;
      *(_DWORD *)(a1 + 8) = v53 + v13 - v12;
LABEL_7:
      if ( *(unsigned int *)(a1 + 12) < v11 )
      {
LABEL_8:
        *(_DWORD *)(a1 + 8) = 0;
        v62 = v11;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v11, 4u, v11, a6);
        v15 = *(unsigned int **)a1;
        v11 = v62;
        v16 = *(_QWORD *)a1 + 4 * v62;
        do
          *v15++ = v10;
        while ( (unsigned int *)v16 != v15 );
        goto LABEL_10;
      }
LABEL_50:
      v39 = *(unsigned int *)(a1 + 8);
      v40 = v11;
      if ( v39 <= v11 )
        v40 = *(unsigned int *)(a1 + 8);
      if ( v40 )
      {
        v41 = *(unsigned int **)a1;
        v42 = *(_QWORD *)a1 + 4 * v40;
        do
          *v41++ = v10;
        while ( (unsigned int *)v42 != v41 );
        v39 = *(unsigned int *)(a1 + 8);
      }
      if ( v39 < v11 )
      {
        v43 = v11 - v39;
        if ( v11 != v39 )
        {
          v44 = (unsigned int *)(*(_QWORD *)a1 + 4 * v39);
          for ( i = &v44[v43]; i != v44; ++v44 )
            *v44 = v10;
        }
      }
LABEL_10:
      *(_DWORD *)(a1 + 8) = v6;
      v17 = 4 * v11;
      v18 = 0;
      if ( !v6 )
        goto LABEL_62;
      do
      {
        v19 = *(int *)(a2 + v18);
        if ( (_DWORD)v19 != -1 )
          *(_DWORD *)(*(_QWORD *)a1 + v18) = v67[v19];
        v18 += 4;
      }
      while ( v17 != v18 );
      v20 = *(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 == v20 )
      {
LABEL_62:
        *(_DWORD *)(a1 + 8) = 0;
      }
      else
      {
        v21 = *(unsigned int **)a1;
        v22 = 0;
        while ( 1 )
        {
          v23 = *v21;
          if ( v10 != (_DWORD)v23 && v23 != v22 )
            break;
          ++v21;
          ++v22;
          if ( (unsigned int *)v20 == v21 )
            goto LABEL_62;
        }
        sub_2B23E00(*(_QWORD *)a1, *(_DWORD *)(a1 + 8));
      }
LABEL_19:
      v24 = v67;
      if ( v67 == (int *)v69 )
        return;
      goto LABEL_20;
    }
    if ( !(_DWORD)a3 )
      goto LABEL_50;
    if ( (unsigned int)a3 > 0xCuLL )
    {
      v66 = (unsigned int)a3;
      sub_C8D5F0((__int64)&v67, v69, (unsigned int)a3, 4u, (unsigned int)a3, a6);
      v11 = v66;
      v34 = v67;
      v31 = v66;
      v32 = &v67[(unsigned int)v68];
      v33 = &v67[v66];
      if ( v33 == v32 )
        goto LABEL_46;
    }
    else
    {
      v31 = (unsigned int)a3;
      v32 = (int *)v69;
      v33 = (int *)&v69[v31 * 4];
    }
    do
    {
      if ( v32 )
        *v32 = 0;
      ++v32;
    }
    while ( v32 != v33 );
    v34 = v67;
    v32 = &v67[v31];
LABEL_46:
    LODWORD(v68) = v6;
    if ( v34 != v32 )
    {
      v35 = v32 - 1;
      v36 = 0;
      v37 = (unsigned __int64)((char *)v35 - (char *)v34) >> 2;
      do
      {
        v38 = v36;
        v34[v36] = v36;
        ++v36;
      }
      while ( v37 != v38 );
      if ( *(unsigned int *)(a1 + 12) < v11 )
        goto LABEL_8;
      goto LABEL_50;
    }
    goto LABEL_7;
  }
  v67 = (int *)v69;
  v68 = 0xC00000000LL;
  if ( v9 )
  {
    sub_2B0FC00(*(_QWORD *)a1, v9, (__int64)&v67, a4, a5, a6);
  }
  else
  {
    a3 = (unsigned int)a3;
    if ( (_DWORD)a3 )
    {
      v28 = (int *)v69;
      v29 = (int *)v69;
      if ( (unsigned int)a3 > 0xCuLL )
      {
        v65 = (unsigned int)a3;
        sub_C8D5F0((__int64)&v67, v69, (unsigned int)a3, 4u, a5, a6);
        v29 = v67;
        a3 = v65;
        v28 = &v67[(unsigned int)v68];
      }
      a4 = 4 * a3;
      a3 = (__int64)&v29[a3];
      if ( (int *)a3 != v28 )
      {
        do
        {
          if ( v28 )
            *v28 = 0;
          ++v28;
        }
        while ( (int *)a3 != v28 );
        v29 = v67;
      }
      LODWORD(v68) = v6;
      if ( a4 )
      {
        v30 = 0;
        a4 = (a4 - 4) >> 2;
        do
        {
          a3 = v30;
          v29[v30] = v30;
          ++v30;
        }
        while ( a4 != a3 );
      }
    }
  }
  sub_2B32FB0((__int64)&v67, a2, a3, a4);
  if ( !(unsigned __int8)sub_B4ED80(v67, (unsigned int)v68, v6) )
  {
    sub_2B39CB0(a1, v6, v6, v25, v26, v27);
    v46 = 0;
    if ( v6 )
    {
      do
      {
        v47 = v67[v46];
        if ( (_DWORD)v47 != -1 )
          *(_DWORD *)(*(_QWORD *)a1 + 4 * v47) = v46;
        ++v46;
      }
      while ( v46 != v6 );
    }
    sub_2B23E00(*(_QWORD *)a1, *(_DWORD *)(a1 + 8));
    goto LABEL_19;
  }
  v24 = v67;
  *(_DWORD *)(a1 + 8) = 0;
  if ( v24 != (int *)v69 )
LABEL_20:
    _libc_free((unsigned __int64)v24);
}
