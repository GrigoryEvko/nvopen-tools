// Function: sub_25C5120
// Address: 0x25c5120
//
__int64 __fastcall sub_25C5120(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r8
  int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rdx
  char v15; // r14
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r14
  __int64 *v21; // r15
  __int64 *v22; // r13
  unsigned __int64 v24; // rcx
  __int64 v25; // r15
  int v26; // eax
  unsigned int v27; // r13d
  char v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 *v34; // rcx
  __int64 v35; // r10
  __int64 v36; // rdx
  int v37; // ecx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 *v40; // rax
  __int64 *v41; // r13
  int v42; // ebx
  char v43; // al
  __int64 *v44; // rax
  __int64 v45; // r14
  __int64 *v46; // r15
  unsigned __int64 v47; // rax
  __int64 *v48; // r14
  __int64 v49; // rdi
  __int64 *v50; // rdx
  __int64 *i; // rbx
  __int64 v52; // rax
  __int64 *v53; // rax
  __int64 v54; // r14
  __int64 v55; // r15
  __int64 v56; // rsi
  __int64 v57; // r14
  __int64 v58; // r13
  int v59; // r10d
  int v60; // r11d
  __int64 v61; // [rsp+8h] [rbp-168h]
  __int64 v62; // [rsp+10h] [rbp-160h]
  int v63; // [rsp+10h] [rbp-160h]
  int v64; // [rsp+18h] [rbp-158h]
  __int64 v65[2]; // [rsp+20h] [rbp-150h] BYREF
  char v66; // [rsp+30h] [rbp-140h] BYREF
  __int64 *v67; // [rsp+70h] [rbp-100h] BYREF
  __int64 v68; // [rsp+78h] [rbp-F8h]
  _BYTE v69[240]; // [rsp+80h] [rbp-F0h] BYREF

  v9 = *a2;
  v10 = *(_BYTE *)(*a2 + 8) & 1;
  if ( v10 )
  {
    v11 = v9 + 16;
    a6 = 15;
  }
  else
  {
    v36 = *(unsigned int *)(v9 + 24);
    v11 = *(_QWORD *)(v9 + 16);
    if ( !(_DWORD)v36 )
      goto LABEL_85;
    a6 = (unsigned int)(v36 - 1);
  }
  v12 = a6 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v62 = v11 + 416LL * v12;
  v13 = *(_QWORD *)v62;
  if ( a3 != *(_QWORD *)v62 )
  {
    v59 = 1;
    while ( v13 != -4096 )
    {
      v12 = a6 & (v12 + v59);
      v62 = v11 + 416LL * v12;
      v13 = *(_QWORD *)v62;
      if ( a3 == *(_QWORD *)v62 )
        goto LABEL_4;
      ++v59;
    }
    if ( (_BYTE)v10 )
    {
      v58 = 6656;
      goto LABEL_86;
    }
    v36 = *(unsigned int *)(v9 + 24);
LABEL_85:
    v58 = 416 * v36;
LABEL_86:
    v62 = v58 + v11;
  }
LABEL_4:
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  v14 = *a2;
  v15 = *(_BYTE *)(*a2 + 8) & 1;
  if ( v15 )
  {
    v16 = v14 + 16;
    v17 = 6656;
  }
  else
  {
    v16 = *(_QWORD *)(v14 + 16);
    v17 = 416LL * *(unsigned int *)(v14 + 24);
  }
  if ( v62 == v17 + v16 || !*(_BYTE *)(v62 + 409) )
  {
    v24 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v24 != a3 + 48 )
    {
      if ( !v24 )
        BUG();
      v25 = v24 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 <= 0xA )
      {
        v61 = *a2;
        v26 = sub_B46E30(v24 - 24);
        v14 = v61;
        v64 = v26;
        if ( v26 )
        {
          v27 = 0;
          v28 = 0;
          do
          {
            v29 = sub_B46EC0(v25, v27);
            v30 = a2[1];
            v31 = *(_QWORD *)(v30 + 8);
            v32 = *(unsigned int *)(v30 + 24);
            if ( !(_DWORD)v32 )
              goto LABEL_41;
            v33 = (v32 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v34 = (__int64 *)(v31 + 88LL * v33);
            v35 = *v34;
            if ( v29 != *v34 )
            {
              v37 = 1;
              while ( v35 != -4096 )
              {
                v60 = v37 + 1;
                v33 = (v32 - 1) & (v37 + v33);
                v34 = (__int64 *)(v31 + 88LL * v33);
                v35 = *v34;
                if ( v29 == *v34 )
                  goto LABEL_30;
                v37 = v60;
              }
LABEL_41:
              v67 = (__int64 *)v69;
              v68 = 0x200000000LL;
              sub_25C2C90(a1, (__int64 *)&v67);
              sub_25C0430((__int64)&v67);
              v14 = *a2;
              v15 = *(_BYTE *)(*a2 + 8) & 1;
              goto LABEL_8;
            }
LABEL_30:
            if ( v34 == (__int64 *)(v31 + 88 * v32) )
              goto LABEL_41;
            if ( v28 )
            {
              sub_ABFB50(&v67, a1, (__int64)(v34 + 1));
              sub_25C2C90(a1, (__int64 *)&v67);
              sub_25C0430((__int64)&v67);
            }
            else
            {
              sub_25C2990(a1, (__int64)(v34 + 1));
            }
            ++v27;
            v28 = 1;
          }
          while ( v64 != v27 );
          v14 = *a2;
          v15 = *(_BYTE *)(*a2 + 8) & 1;
        }
      }
    }
  }
LABEL_8:
  if ( v15 )
  {
    v18 = v14 + 16;
    v19 = 6656;
  }
  else
  {
    v18 = *(_QWORD *)(v14 + 16);
    v19 = 416LL * *(unsigned int *)(v14 + 24);
  }
  if ( v62 == v19 + v18 )
    return a1;
  v67 = (__int64 *)v69;
  v68 = 0x200000000LL;
  if ( (*(_BYTE *)(v62 + 16) & 1) != 0 )
  {
    v21 = (__int64 *)(v62 + 408);
    if ( !(*(_DWORD *)(v62 + 16) >> 1) )
      goto LABEL_17;
    v20 = (__int64 *)(v62 + 24);
  }
  else
  {
    v20 = *(__int64 **)(v62 + 24);
    v21 = &v20[12 * *(unsigned int *)(v62 + 32)];
    if ( !(*(_DWORD *)(v62 + 16) >> 1) )
      goto LABEL_17;
  }
  if ( v20 == v21 )
    goto LABEL_17;
  while ( *v20 == -4096 || *v20 == -8192 )
  {
    v20 += 12;
    if ( v20 == v21 )
      goto LABEL_17;
  }
  if ( v20 == v21 )
  {
LABEL_17:
    v22 = (__int64 *)v69;
    goto LABEL_18;
  }
  v38 = (__int64)v20;
  v39 = 0;
  while ( 1 )
  {
    v40 = (__int64 *)(v38 + 96);
    if ( (__int64 *)(v38 + 96) == v21 )
      break;
    while ( 1 )
    {
      v38 = (__int64)v40;
      if ( *v40 != -8192 && *v40 != -4096 )
        break;
      v40 += 12;
      if ( v21 == v40 )
        goto LABEL_48;
    }
    ++v39;
    if ( v40 == v21 )
      goto LABEL_49;
  }
LABEL_48:
  ++v39;
LABEL_49:
  v41 = (__int64 *)v69;
  v42 = v39;
  if ( v39 > 2 )
  {
    v63 = v39;
    sub_25C30E0((__int64)&v67, v39, v38, v39, v9, a6);
    v42 = v63;
    v41 = &v67[12 * (unsigned int)v68];
  }
  do
  {
    if ( v41 )
    {
      *v41 = *v20;
      v43 = *((_BYTE *)v20 + 8);
      *((_DWORD *)v41 + 6) = 0;
      *((_BYTE *)v41 + 8) = v43;
      v41[2] = (__int64)(v41 + 4);
      *((_DWORD *)v41 + 7) = 2;
      if ( *((_DWORD *)v20 + 6) )
        sub_25C2990((__int64)(v41 + 2), (__int64)(v20 + 2));
    }
    v44 = v20 + 12;
    if ( v20 + 12 == v21 )
      break;
    while ( 1 )
    {
      v20 = v44;
      if ( *v44 != -8192 && *v44 != -4096 )
        break;
      v44 += 12;
      if ( v21 == v44 )
        goto LABEL_58;
    }
    v41 += 12;
  }
  while ( v44 != v21 );
LABEL_58:
  v22 = v67;
  LODWORD(v68) = v68 + v42;
  v45 = 12LL * (unsigned int)v68;
  v46 = &v67[v45];
  if ( &v67[v45] != v67 )
  {
    _BitScanReverse64(&v47, 0xAAAAAAAAAAAAAAABLL * ((v45 * 8) >> 5));
    sub_25C4870((__int64)v67, (unsigned __int64)&v67[v45], 2LL * (int)(63 - (v47 ^ 0x3F)));
    if ( (unsigned __int64)v45 <= 192 )
    {
      sub_25C34B0((__int64)v22, v46);
    }
    else
    {
      v48 = v22 + 192;
      sub_25C34B0((__int64)v22, v22 + 192);
      if ( v46 != v22 + 192 )
      {
        do
        {
          v49 = (__int64)v48;
          v48 += 12;
          sub_25C3380(v49);
        }
        while ( v46 != v48 );
      }
    }
    v22 = v67;
    v50 = &v67[12 * (unsigned int)v68];
    if ( v50 != v67 )
    {
      for ( i = v50 - 10; ; i = v53 )
      {
        if ( (*(_BYTE *)(i - 1) & 0xFD) == 1 )
        {
          v65[0] = (__int64)&v66;
          v65[1] = 0x200000000LL;
          sub_25C2C90(a1, v65);
          sub_25C0430((__int64)v65);
        }
        v52 = *((unsigned int *)i + 2);
        if ( !(_DWORD)v52 )
          goto LABEL_67;
        if ( *((_BYTE *)i - 8) <= 1u )
          break;
        v54 = *i;
        v55 = *i + 32 * v52;
        do
        {
          v56 = v54;
          v54 += 32;
          sub_AC04E0(a1, v56);
        }
        while ( v55 != v54 );
        v53 = i - 12;
        if ( v22 == i - 2 )
        {
LABEL_80:
          v57 = (__int64)v67;
          v22 = &v67[12 * (unsigned int)v68];
          if ( v67 != v22 )
          {
            do
            {
              v22 -= 12;
              sub_25C0430((__int64)(v22 + 2));
            }
            while ( (__int64 *)v57 != v22 );
            v22 = v67;
          }
          goto LABEL_18;
        }
LABEL_68:
        ;
      }
      sub_AC10C0(v65, (unsigned int *)a1, i);
      sub_25C2C90(a1, v65);
      sub_25C0430((__int64)v65);
LABEL_67:
      v53 = i - 12;
      if ( v22 == i - 2 )
        goto LABEL_80;
      goto LABEL_68;
    }
  }
LABEL_18:
  if ( v22 != (__int64 *)v69 )
    _libc_free((unsigned __int64)v22);
  return a1;
}
