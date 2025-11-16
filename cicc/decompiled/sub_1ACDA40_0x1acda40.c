// Function: sub_1ACDA40
// Address: 0x1acda40
//
__int64 __fastcall sub_1ACDA40(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *m; // rdx
  unsigned int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  _QWORD *v17; // rsi
  __int64 v18; // rdx
  unsigned int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // r14
  __int64 v22; // r12
  unsigned int v23; // r13d
  unsigned int v24; // r15d
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // r14
  int v27; // r12d
  char v28; // dl
  __int64 v29; // r8
  int v30; // r9d
  __int64 v31; // rax
  __int64 v32; // r8
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 *v36; // rax
  __int64 *v37; // rdi
  __int64 *v38; // rcx
  unsigned int v40; // ecx
  _QWORD *v41; // rdi
  unsigned int v42; // eax
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rax
  int v46; // ebx
  __int64 v47; // r12
  _QWORD *v48; // rax
  __int64 v49; // rdx
  _QWORD *k; // rdx
  unsigned int v51; // ecx
  _QWORD *v52; // rdi
  unsigned int v53; // eax
  int v54; // eax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  int v57; // ebx
  __int64 v58; // r12
  _QWORD *v59; // rax
  __int64 v60; // rdx
  _QWORD *j; // rdx
  _QWORD *v62; // rax
  _QWORD *v63; // rax
  __int64 v64; // [rsp+20h] [rbp-210h]
  __int64 v65; // [rsp+20h] [rbp-210h]
  __int64 *v66; // [rsp+28h] [rbp-208h]
  _QWORD *v67; // [rsp+30h] [rbp-200h] BYREF
  unsigned int v68; // [rsp+38h] [rbp-1F8h]
  unsigned int v69; // [rsp+3Ch] [rbp-1F4h]
  _QWORD v70[8]; // [rsp+40h] [rbp-1F0h] BYREF
  _QWORD *v71; // [rsp+80h] [rbp-1B0h] BYREF
  unsigned int v72; // [rsp+88h] [rbp-1A8h]
  unsigned int v73; // [rsp+8Ch] [rbp-1A4h]
  _QWORD v74[8]; // [rsp+90h] [rbp-1A0h] BYREF
  __int64 v75; // [rsp+D0h] [rbp-160h] BYREF
  __int64 *v76; // [rsp+D8h] [rbp-158h]
  __int64 *v77; // [rsp+E0h] [rbp-150h]
  unsigned int v78; // [rsp+E8h] [rbp-148h]
  unsigned int v79; // [rsp+ECh] [rbp-144h]
  int v80; // [rsp+F0h] [rbp-140h]
  _QWORD v81[39]; // [rsp+F8h] [rbp-138h] BYREF

  v2 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 16);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 36) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 40);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 24));
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_DWORD *)(a1 + 40) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v51 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v51 = 64;
  if ( (unsigned int)v3 <= v51 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 24);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 32) = 0;
    goto LABEL_7;
  }
  v52 = *(_QWORD **)(a1 + 24);
  v53 = v2 - 1;
  if ( !v53 )
  {
    v58 = 2048;
    v57 = 128;
LABEL_71:
    j___libc_free_0(v52);
    *(_DWORD *)(a1 + 40) = v57;
    v59 = (_QWORD *)sub_22077B0(v58);
    v60 = *(unsigned int *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 24) = v59;
    for ( j = &v59[2 * v60]; j != v59; v59 += 2 )
    {
      if ( v59 )
        *v59 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v53, v53);
  v54 = 1 << (33 - (v53 ^ 0x1F));
  if ( v54 < 64 )
    v54 = 64;
  if ( (_DWORD)v3 != v54 )
  {
    v55 = (4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1);
    v56 = ((v55 | (v55 >> 2)) >> 4) | v55 | (v55 >> 2) | ((((v55 | (v55 >> 2)) >> 4) | v55 | (v55 >> 2)) >> 8);
    v57 = (v56 | (v56 >> 16)) + 1;
    v58 = 16 * ((v56 | (v56 >> 16)) + 1);
    goto LABEL_71;
  }
  *(_QWORD *)(a1 + 32) = 0;
  v62 = &v52[2 * (unsigned int)v3];
  do
  {
    if ( v52 )
      *v52 = -8;
    v52 += 2;
  }
  while ( v62 != v52 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  if ( v6 )
  {
    v40 = 4 * v6;
    v7 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)(4 * v6) < 0x40 )
      v40 = 64;
    if ( (unsigned int)v7 <= v40 )
      goto LABEL_10;
    v41 = *(_QWORD **)(a1 + 56);
    v42 = v6 - 1;
    if ( v42 )
    {
      _BitScanReverse(&v42, v42);
      v43 = (unsigned int)(1 << (33 - (v42 ^ 0x1F)));
      if ( (int)v43 < 64 )
        v43 = 64;
      if ( (_DWORD)v43 == (_DWORD)v7 )
      {
        *(_QWORD *)(a1 + 64) = 0;
        v63 = &v41[2 * v43];
        do
        {
          if ( v41 )
            *v41 = -8;
          v41 += 2;
        }
        while ( v63 != v41 );
        goto LABEL_13;
      }
      v44 = (4 * (int)v43 / 3u + 1) | ((unsigned __int64)(4 * (int)v43 / 3u + 1) >> 1);
      v45 = ((v44 | (v44 >> 2)) >> 4) | v44 | (v44 >> 2) | ((((v44 | (v44 >> 2)) >> 4) | v44 | (v44 >> 2)) >> 8);
      v46 = (v45 | (v45 >> 16)) + 1;
      v47 = 16 * ((v45 | (v45 >> 16)) + 1);
    }
    else
    {
      v47 = 2048;
      v46 = 128;
    }
    j___libc_free_0(v41);
    *(_DWORD *)(a1 + 72) = v46;
    v48 = (_QWORD *)sub_22077B0(v47);
    v49 = *(unsigned int *)(a1 + 72);
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 56) = v48;
    for ( k = &v48[2 * v49]; k != v48; v48 += 2 )
    {
      if ( v48 )
        *v48 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 68) )
  {
    v7 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)v7 <= 0x40 )
    {
LABEL_10:
      v8 = *(_QWORD **)(a1 + 56);
      for ( m = &v8[2 * v7]; m != v8; v8 += 2 )
        *v8 = -8;
      *(_QWORD *)(a1 + 64) = 0;
      goto LABEL_13;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 56));
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = 0;
    *(_DWORD *)(a1 + 72) = 0;
  }
LABEL_13:
  v10 = sub_1ACD7C0((__int64 *)a1);
  if ( v10 )
    return v10;
  v69 = 8;
  v67 = v70;
  v71 = v74;
  v76 = v81;
  v77 = v81;
  v11 = *(_QWORD *)a1;
  v73 = 8;
  v78 = 32;
  v12 = *(_QWORD *)(v11 + 80);
  v80 = 0;
  if ( v12 )
  {
    v12 -= 24;
    v13 = v12;
  }
  else
  {
    v13 = 0;
  }
  v70[0] = v13;
  v14 = *(_QWORD *)(a1 + 8);
  v68 = 1;
  v15 = v70;
  v16 = *(_QWORD *)(v14 + 80);
  v72 = 1;
  v17 = v74;
  v18 = v16 - 24;
  v79 = 1;
  if ( !v16 )
    v18 = 0;
  v66 = (__int64 *)a1;
  v75 = 1;
  v74[0] = v18;
  v19 = 1;
  v81[0] = v12;
  v20 = 1;
  while ( 1 )
  {
    v21 = v15[v20 - 1];
    v68 = v20 - 1;
    v22 = v17[v19 - 1];
    v72 = v19 - 1;
    v23 = sub_1ACCBA0((__int64)v66, v21, v22);
    if ( v23 )
      break;
    v23 = sub_1ACD650(v66, v21, v22);
    if ( v23 )
      break;
    v24 = 0;
    v25 = sub_157EBA0(v21);
    v26 = sub_157EBA0(v22);
    v27 = sub_15F4D60(v25);
    if ( v27 )
    {
      while ( 1 )
      {
        v35 = sub_15F4DF0(v25, v24);
        v36 = v76;
        if ( v77 == v76 )
        {
          v37 = &v76[v79];
          if ( v76 != v37 )
          {
            v38 = 0;
            while ( v35 != *v36 )
            {
              if ( *v36 == -2 )
                v38 = v36;
              if ( v37 == ++v36 )
              {
                if ( !v38 )
                  goto LABEL_41;
                *v38 = v35;
                --v80;
                ++v75;
                goto LABEL_24;
              }
            }
            goto LABEL_29;
          }
LABEL_41:
          if ( v79 < v78 )
            break;
        }
        sub_16CCBA0((__int64)&v75, v35);
        if ( v28 )
          goto LABEL_24;
LABEL_29:
        if ( v27 == ++v24 )
          goto LABEL_39;
      }
      ++v79;
      *v37 = v35;
      ++v75;
LABEL_24:
      v29 = sub_15F4DF0(v25, v24);
      v31 = v68;
      if ( v68 >= v69 )
      {
        v65 = v29;
        sub_16CD150((__int64)&v67, v70, 0, 8, v29, v30);
        v31 = v68;
        v29 = v65;
      }
      v67[v31] = v29;
      ++v68;
      v32 = sub_15F4DF0(v26, v24);
      v34 = v72;
      if ( v72 >= v73 )
      {
        v64 = v32;
        sub_16CD150((__int64)&v71, v74, 0, 8, v32, v33);
        v34 = v72;
        v32 = v64;
      }
      v71[v34] = v32;
      ++v72;
      goto LABEL_29;
    }
LABEL_39:
    v20 = v68;
    if ( !v68 )
      break;
    v17 = v71;
    v19 = v72;
    v15 = v67;
  }
  v10 = v23;
  if ( v77 != v76 )
    _libc_free((unsigned __int64)v77);
  if ( v71 != v74 )
    _libc_free((unsigned __int64)v71);
  if ( v67 != v70 )
    _libc_free((unsigned __int64)v67);
  return v10;
}
