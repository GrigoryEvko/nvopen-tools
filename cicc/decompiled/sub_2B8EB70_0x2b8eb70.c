// Function: sub_2B8EB70
// Address: 0x2b8eb70
//
__int64 __fastcall sub_2B8EB70(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  char *v7; // r10
  unsigned __int64 v8; // r8
  __int64 v9; // r9
  char *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  char *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r14
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  char *v24; // r14
  signed int v25; // r8d
  unsigned int v26; // edx
  unsigned int v27; // eax
  char *v28; // rax
  char *v29; // rsi
  int v30; // edx
  int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // r10d
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // ecx
  char *v42; // rcx
  __int64 v43; // rax
  unsigned __int64 v44; // rsi
  char *v45; // r14
  _QWORD *v46; // rax
  _QWORD *v47; // rcx
  __int64 v48; // rsi
  __int64 v49; // rcx
  char *v50; // [rsp+40h] [rbp-250h]
  __int64 v51; // [rsp+48h] [rbp-248h]
  int v54; // [rsp+58h] [rbp-238h]
  int v55; // [rsp+58h] [rbp-238h]
  unsigned int v56; // [rsp+58h] [rbp-238h]
  unsigned int v57; // [rsp+60h] [rbp-230h] BYREF
  signed int v58; // [rsp+64h] [rbp-22Ch] BYREF
  unsigned __int64 v59; // [rsp+68h] [rbp-228h] BYREF
  _QWORD v60[4]; // [rsp+70h] [rbp-220h] BYREF
  char *v61; // [rsp+90h] [rbp-200h] BYREF
  __int64 v62; // [rsp+98h] [rbp-1F8h]
  _BYTE v63[16]; // [rsp+A0h] [rbp-1F0h] BYREF
  char **v64; // [rsp+B0h] [rbp-1E0h] BYREF
  _QWORD *v65; // [rsp+B8h] [rbp-1D8h]
  signed int *v66; // [rsp+C0h] [rbp-1D0h]
  _DWORD **v67; // [rsp+C8h] [rbp-1C8h]
  char *v68; // [rsp+D0h] [rbp-1C0h] BYREF
  __int64 v69; // [rsp+D8h] [rbp-1B8h]
  _BYTE dest[48]; // [rsp+E0h] [rbp-1B0h] BYREF
  _DWORD *v71; // [rsp+110h] [rbp-180h] BYREF
  __int64 v72; // [rsp+118h] [rbp-178h]
  _BYTE v73[48]; // [rsp+120h] [rbp-170h] BYREF
  _DWORD *v74; // [rsp+150h] [rbp-140h] BYREF
  __int64 v75; // [rsp+158h] [rbp-138h]
  _BYTE v76[48]; // [rsp+160h] [rbp-130h] BYREF
  char *v77; // [rsp+190h] [rbp-100h] BYREF
  int v78; // [rsp+198h] [rbp-F8h]
  char v79; // [rsp+1A0h] [rbp-F0h] BYREF
  char *v80; // [rsp+1D0h] [rbp-C0h] BYREF
  int v81; // [rsp+1D8h] [rbp-B8h]
  char v82; // [rsp+1E0h] [rbp-B0h] BYREF
  __int64 **v83; // [rsp+210h] [rbp-80h] BYREF
  __int64 v84; // [rsp+218h] [rbp-78h]
  _BYTE v85[112]; // [rsp+220h] [rbp-70h] BYREF

  v7 = *(char **)a3;
  v8 = *(unsigned int *)(a3 + 8);
  v68 = dest;
  v69 = 0x600000000LL;
  v9 = 8 * v8;
  if ( v8 > 6 )
  {
    v50 = v7;
    v51 = 8 * v8;
    v55 = v8;
    sub_C8D5F0((__int64)&v68, dest, v8, 8u, v8, v9);
    LODWORD(v8) = v55;
    v9 = v51;
    v7 = v50;
    v14 = &v68[8 * (unsigned int)v69];
  }
  else
  {
    v10 = dest;
    if ( !v9 )
      goto LABEL_3;
    v14 = dest;
  }
  v54 = v8;
  memcpy(v14, v7, v9);
  v10 = v68;
  LODWORD(v9) = v69;
  LODWORD(v8) = v54;
LABEL_3:
  LODWORD(v69) = v9 + v8;
  v11 = *(_QWORD *)v10;
  v57 = v9 + v8;
  v12 = *(_QWORD *)(v11 + 8);
  if ( !sub_2B08630(v12) )
  {
    *(_BYTE *)(a1 + 32) = 0;
    goto LABEL_5;
  }
  v15 = sub_2B08680(v12, v57);
  v56 = sub_2B1F810(*(_QWORD *)(a2 + 3296), v15, v57);
  v74 = v76;
  v71 = v73;
  v83 = (__int64 **)v85;
  v72 = 0xC00000000LL;
  v75 = 0xC00000000LL;
  v84 = 0x100000000LL;
  sub_2B52BE0(&v77, a2, (__int64 *)&v68, (__int64)&v71, v56, (__int64)&v74);
  sub_2B8E0E0(&v80, a2, a3, v68, (unsigned int)v69, (__int64)&v74, (__int64)&v83, v56, 1);
  if ( v78 | v81 )
  {
    v61 = v63;
    v62 = 0x400000000LL;
    sub_2B39CB0((__int64)&v61, (int)v57, v57, v16, v17, v18);
    if ( v81 == 1 && *(_DWORD *)v80 == 7 && sub_2B31C30(**v83, *(char **)a3, *(unsigned int *)(a3 + 8), v21, v22, v23) )
    {
      if ( !a4 )
      {
        v39 = **v83;
        if ( *(_QWORD *)(v39 + 184) != *(_QWORD *)(a3 + 184) && (a5 || *(_DWORD *)(v39 + 200)) )
        {
          v40 = *(unsigned int *)(v39 + 120);
          if ( (_DWORD)v40 )
          {
            v41 = *(_DWORD *)(a3 + 120);
            if ( !v41 )
              v41 = *(_DWORD *)(a3 + 8);
            if ( v41 == 2 && (_DWORD)v75 == 2 )
            {
              v48 = *(_QWORD *)(v39 + 112);
              v49 = 0;
              v37 = v48 + 4 * v40;
              while ( v37 != v48 + 4 * v49 )
              {
                v39 = *(_DWORD *)(v48 + 4 * v49) % 2;
                if ( v39 != (v49 & 1) )
                  goto LABEL_67;
                ++v49;
              }
            }
          }
          v42 = v61;
          if ( v61 != &v61[4 * (unsigned int)v62] )
          {
            v43 = 0;
            v44 = (4 * (unsigned __int64)(unsigned int)v62 - 4) >> 2;
            do
            {
              v39 = v43;
              *(_DWORD *)&v42[4 * v43] = v43;
              ++v43;
            }
            while ( v39 != v44 );
          }
          sub_2B39C80(a1, &v61, v39, (__int64)v42, v37, v38);
          *(_BYTE *)(a1 + 32) = 1;
          goto LABEL_68;
        }
      }
      goto LABEL_67;
    }
    if ( !v78
      && (LODWORD(v64) = -1, v45 = (char *)&v74[(unsigned int)v75], v45 == sub_2B14AB0(v74, v45, (int *)&v64))
      && ((_DWORD)v84 != 1 || !*(_DWORD *)(**v83 + 152))
      || !v81 && (LODWORD(v64) = -1, v24 = (char *)&v71[(unsigned int)v72], v24 == sub_2B14AB0(v71, v24, (int *)&v64)) )
    {
LABEL_67:
      *(_BYTE *)(a1 + 32) = 0;
      goto LABEL_68;
    }
    sub_B48880((__int64 *)&v59, v56, 0);
    v25 = v57;
    v60[1] = &v57;
    v60[0] = &v59;
    v60[2] = &v68;
    v26 = 1;
    v27 = (v57 != 0) + (v57 - (v57 != 0)) / v56;
    if ( v27 > 1 )
    {
      _BitScanReverse(&v27, v27 - 1);
      v26 = 1 << (32 - (v27 ^ 0x1F));
    }
    if ( v57 > v26 )
      v25 = v26;
    v58 = v25;
    if ( v78 )
    {
      v64 = &v77;
      v66 = &v58;
      v65 = (_QWORD *)a3;
      v67 = &v71;
      sub_2B1EA40(
        (__int64)v60,
        (__int64)v61,
        v62,
        (__int64)v71,
        v25,
        v56,
        (__int64 (__fastcall *)(__int64, _QWORD))sub_2B0C5A0,
        (__int64)&v64);
    }
    if ( v81 == 1 && v56 != 1 )
    {
      if ( (v59 & 1) != 0 )
      {
        if ( ((v59 >> 1) & ~(-1LL << (v59 >> 58))) != 0 )
          goto LABEL_71;
      }
      else
      {
        v46 = sub_2B0B280(*(_QWORD **)v59, *(_QWORD *)v59 + 8LL * *(unsigned int *)(v59 + 8));
        if ( v47 != v46 )
          goto LABEL_71;
      }
      v56 = 1;
      v58 = v57;
    }
    if ( (_DWORD)v84 )
    {
      v64 = &v80;
      v65 = &v83;
      sub_2B1EA40(
        (__int64)v60,
        (__int64)v61,
        v62,
        (__int64)v74,
        v58,
        v56,
        (__int64 (__fastcall *)(__int64, _QWORD))sub_2B085A0,
        (__int64)&v64);
    }
    v28 = v61;
    v29 = &v61[4 * (unsigned int)v62];
    if ( v61 != v29 )
    {
      v30 = 0;
      do
      {
        v31 = *(_DWORD *)v28 == v57;
        v28 += 4;
        v30 += v31;
      }
      while ( v29 != v28 );
    }
    if ( !(unsigned __int8)sub_2B0D9E0(v59) && ((int)v57 <= 2 || (int)v57 >> 1 > (int)v35) )
    {
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 0x400000000LL;
      if ( v36 )
        sub_2B0D510(a1, &v61, v32, v33, v34, v35);
      *(_BYTE *)(a1 + 32) = 1;
      goto LABEL_72;
    }
LABEL_71:
    *(_BYTE *)(a1 + 32) = 0;
LABEL_72:
    sub_228BF40((unsigned __int64 **)&v59);
LABEL_68:
    if ( v61 != v63 )
      _libc_free((unsigned __int64)v61);
    goto LABEL_13;
  }
  *(_BYTE *)(a1 + 32) = 0;
LABEL_13:
  if ( v80 != &v82 )
    _libc_free((unsigned __int64)v80);
  if ( v77 != &v79 )
    _libc_free((unsigned __int64)v77);
  v19 = (unsigned __int64 *)v83;
  v20 = (unsigned __int64 *)&v83[8 * (unsigned __int64)(unsigned int)v84];
  if ( v83 != (__int64 **)v20 )
  {
    do
    {
      v20 -= 8;
      if ( (unsigned __int64 *)*v20 != v20 + 2 )
        _libc_free(*v20);
    }
    while ( v19 != v20 );
    v20 = (unsigned __int64 *)v83;
  }
  if ( v20 != (unsigned __int64 *)v85 )
    _libc_free((unsigned __int64)v20);
  if ( v74 != (_DWORD *)v76 )
    _libc_free((unsigned __int64)v74);
  if ( v71 != (_DWORD *)v73 )
    _libc_free((unsigned __int64)v71);
LABEL_5:
  if ( v68 != dest )
    _libc_free((unsigned __int64)v68);
  return a1;
}
