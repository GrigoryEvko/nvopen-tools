// Function: sub_1CA8520
// Address: 0x1ca8520
//
__int64 __fastcall sub_1CA8520(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  int v9; // eax
  _QWORD *v10; // rdx
  _QWORD *v11; // r15
  _QWORD *v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned __int64 *v16; // rdx
  unsigned __int64 *v17; // r12
  unsigned __int64 *v18; // r9
  unsigned __int64 *v19; // rax
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rcx
  unsigned __int64 *v22; // r8
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rax
  unsigned __int64 *v25; // rax
  unsigned __int64 *v26; // r8
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // rdx
  _QWORD *v29; // rdx
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rsi
  _QWORD *v32; // rax
  unsigned __int64 v33; // rax
  bool v34; // cl
  _QWORD *v35; // r13
  __int64 v36; // rcx
  __int64 v37; // rax
  unsigned int v38; // r12d
  __int64 v40; // rax
  _QWORD *v41; // rsi
  _QWORD *v42; // rax
  _QWORD *v43; // rdx
  _BOOL8 v44; // rdi
  _QWORD *v45; // rdi
  unsigned __int64 v46; // rdx
  bool v47; // al
  _QWORD *v48; // r15
  __int64 v49; // rbx
  __int64 v50; // r13
  _BYTE *v51; // rsi
  unsigned int v52; // r8d
  unsigned int v53; // r10d
  __int64 v54; // rdx
  unsigned int v55; // r9d
  unsigned __int64 *v56; // rax
  _BYTE *v57; // rdi
  unsigned __int64 *v58; // rcx
  unsigned __int64 v59; // rcx
  unsigned __int64 *v60; // r11
  int v61; // edi
  int v62; // ecx
  int v63; // edx
  unsigned int v64; // r11d
  int v65; // ecx
  int v66; // esi
  int v67; // [rsp+4h] [rbp-6Ch]
  __int64 v68; // [rsp+8h] [rbp-68h]
  _QWORD *v69; // [rsp+10h] [rbp-60h]
  char v70; // [rsp+10h] [rbp-60h]
  _BYTE *v72; // [rsp+28h] [rbp-48h] BYREF
  _BYTE *v73; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 *v74[7]; // [rsp+38h] [rbp-38h] BYREF

  v72 = *(_BYTE **)(a2 - 24);
  v9 = sub_1CA8350((__int64)a1, v72, a3, a4);
  if ( v9 == 4 )
  {
    if ( byte_4FBE1C0 )
      return 1;
  }
  else if ( v9 == 16 && (unsigned __int8)sub_1C2F070(a4) && unk_4FBE1ED )
  {
    return 1;
  }
  v10 = (_QWORD *)a1[64];
  v11 = a1 + 63;
  v12 = a1 + 63;
  v13 = v10;
  if ( !v10 )
    goto LABEL_10;
  do
  {
    while ( 1 )
    {
      v14 = v13[2];
      v15 = v13[3];
      if ( v13[4] >= a2 )
        break;
      v13 = (_QWORD *)v13[3];
      if ( !v15 )
        goto LABEL_8;
    }
    v12 = v13;
    v13 = (_QWORD *)v13[2];
  }
  while ( v14 );
LABEL_8:
  if ( v11 == v12 || v12[4] > a2 )
  {
LABEL_10:
    v16 = (unsigned __int64 *)a1[40];
    v17 = a1 + 39;
    if ( v16 )
    {
      v18 = a1 + 39;
      v19 = (unsigned __int64 *)a1[40];
      do
      {
        while ( 1 )
        {
          v20 = v19[2];
          v21 = v19[3];
          if ( v19[4] >= (unsigned __int64)v72 )
            break;
          v19 = (unsigned __int64 *)v19[3];
          if ( !v21 )
            goto LABEL_15;
        }
        v18 = v19;
        v19 = (unsigned __int64 *)v19[2];
      }
      while ( v20 );
LABEL_15:
      if ( v17 != v18 && v18[4] <= (unsigned __int64)v72 )
      {
        v22 = a1 + 39;
        v69 = a1 + 33;
        do
        {
          while ( 1 )
          {
            v23 = v16[2];
            v24 = v16[3];
            if ( v16[4] >= (unsigned __int64)v72 )
              break;
            v16 = (unsigned __int64 *)v16[3];
            if ( !v24 )
              goto LABEL_21;
          }
          v22 = v16;
          v16 = (unsigned __int64 *)v16[2];
        }
        while ( v23 );
LABEL_21:
        if ( v17 == v22 || v22[4] > (unsigned __int64)v72 )
        {
          v74[0] = (unsigned __int64 *)&v72;
          v22 = sub_1C9AC70(a1 + 38, v22, v74);
        }
        if ( v69 != sub_1C98E50((__int64)(a1 + 32), v22 + 5) )
        {
          v25 = (unsigned __int64 *)a1[40];
          if ( v25 )
          {
            v26 = a1 + 39;
            do
            {
              while ( 1 )
              {
                v27 = v25[2];
                v28 = v25[3];
                if ( v25[4] >= (unsigned __int64)v72 )
                  break;
                v25 = (unsigned __int64 *)v25[3];
                if ( !v28 )
                  goto LABEL_30;
              }
              v26 = v25;
              v25 = (unsigned __int64 *)v25[2];
            }
            while ( v27 );
LABEL_30:
            if ( v17 != v26 && v26[4] <= (unsigned __int64)v72 )
            {
LABEL_33:
              v29 = (_QWORD *)a1[34];
              if ( v29 )
              {
                v30 = v26[5];
                v31 = a1 + 33;
                while ( 1 )
                {
                  v33 = v29[4];
                  v34 = v33 < v30;
                  if ( v33 == v30 )
                    v34 = v29[5] < v26[6];
                  v32 = (_QWORD *)v29[3];
                  if ( !v34 )
                  {
                    v32 = (_QWORD *)v29[2];
                    v31 = v29;
                  }
                  if ( !v32 )
                    break;
                  v29 = v32;
                }
                if ( v69 != v31 )
                {
                  v46 = v31[4];
                  v47 = v46 > v30;
                  if ( v46 == v30 )
                    v47 = v26[6] < v31[5];
                  if ( !v47 )
                    goto LABEL_74;
                }
              }
              else
              {
                v31 = a1 + 33;
              }
              v74[0] = v26 + 5;
              v31 = (_QWORD *)sub_1C9CF10(a1 + 32, v31, (const __m128i **)v74);
LABEL_74:
              v48 = v31 + 7;
              if ( v31 + 7 == (_QWORD *)v31[9] )
                return 0;
              v70 = 0;
              v38 = 0;
              v68 = (__int64)a1;
              v49 = a3;
              v50 = v31[9];
              while ( 1 )
              {
                v51 = *(_BYTE **)(v50 + 32);
                if ( *(_BYTE *)(*(_QWORD *)v51 + 8LL) != 15 )
                  return 15;
                if ( v51[16] <= 0x17u )
                {
                  v38 |= sub_1CA8350(v68, v51, v49, a4);
                  goto LABEL_84;
                }
                v52 = *(_DWORD *)(v49 + 24);
                if ( !v52 )
                  goto LABEL_83;
                v53 = v52 - 1;
                v54 = *(_QWORD *)(v49 + 8);
                v55 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                v56 = (unsigned __int64 *)(v54 + 16LL * v55);
                v57 = (_BYTE *)*v56;
                v58 = v56;
                if ( v51 != (_BYTE *)*v56 )
                  break;
LABEL_80:
                if ( v58 == (unsigned __int64 *)(v54 + 16LL * v52) )
                  goto LABEL_83;
                v73 = *(_BYTE **)(v50 + 32);
                v59 = *v56;
                if ( v51 != (_BYTE *)*v56 )
                {
                  v60 = 0;
                  v61 = 1;
                  while ( v59 != -8 )
                  {
                    if ( !v60 && v59 == -16 )
                      v60 = v56;
                    v55 = v53 & (v61 + v55);
                    v56 = (unsigned __int64 *)(v54 + 16LL * v55);
                    v59 = *v56;
                    if ( v51 == (_BYTE *)*v56 )
                      goto LABEL_82;
                    ++v61;
                  }
                  v62 = *(_DWORD *)(v49 + 16);
                  if ( v60 )
                    v56 = v60;
                  ++*(_QWORD *)v49;
                  v63 = v62 + 1;
                  if ( 4 * (v62 + 1) >= 3 * v52 )
                  {
                    v66 = 2 * v52;
                  }
                  else
                  {
                    if ( v52 - *(_DWORD *)(v49 + 20) - v63 > v52 >> 3 )
                    {
LABEL_94:
                      *(_DWORD *)(v49 + 16) = v63;
                      if ( *v56 != -8 )
                        --*(_DWORD *)(v49 + 20);
                      *v56 = (unsigned __int64)v51;
                      *((_DWORD *)v56 + 2) = 0;
                      goto LABEL_84;
                    }
                    v66 = v52;
                  }
                  sub_177C7D0(v49, v66);
                  sub_190E590(v49, (__int64 *)&v73, v74);
                  v56 = v74[0];
                  v51 = v73;
                  v63 = *(_DWORD *)(v49 + 16) + 1;
                  goto LABEL_94;
                }
LABEL_82:
                v38 |= *((_DWORD *)v56 + 2);
LABEL_84:
                v50 = sub_220EF30(v50);
                if ( v48 == (_QWORD *)v50 )
                {
                  if ( v38 != 15 && v70 )
                    *a5 = 1;
                  return v38;
                }
              }
              v64 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
              v65 = 1;
              while ( v57 != (_BYTE *)-8LL )
              {
                v64 = v53 & (v65 + v64);
                v67 = v65 + 1;
                v58 = (unsigned __int64 *)(v54 + 16LL * v64);
                v57 = (_BYTE *)*v58;
                if ( v51 == (_BYTE *)*v58 )
                  goto LABEL_80;
                v65 = v67;
              }
LABEL_83:
              v70 = 1;
              goto LABEL_84;
            }
          }
          else
          {
            v26 = a1 + 39;
          }
          v74[0] = (unsigned __int64 *)&v72;
          v26 = sub_1C9AC70(a1 + 38, v26, v74);
          goto LABEL_33;
        }
      }
    }
    return 15;
  }
  v35 = a1 + 63;
  do
  {
    while ( 1 )
    {
      v36 = v10[2];
      v37 = v10[3];
      if ( v10[4] >= a2 )
        break;
      v10 = (_QWORD *)v10[3];
      if ( !v37 )
        goto LABEL_45;
    }
    v35 = v10;
    v10 = (_QWORD *)v10[2];
  }
  while ( v36 );
LABEL_45:
  if ( v11 == v35 || v35[4] > a2 )
  {
    v40 = sub_22077B0(48);
    v41 = v35;
    *(_QWORD *)(v40 + 32) = a2;
    v35 = (_QWORD *)v40;
    *(_DWORD *)(v40 + 40) = 0;
    v42 = sub_1C9D3C0(a1 + 62, v41, (unsigned __int64 *)(v40 + 32));
    if ( v43 )
    {
      v44 = v11 == v43 || v42 || v43[4] > a2;
      sub_220F040(v44, v35, v43, a1 + 63);
      ++a1[67];
    }
    else
    {
      v45 = v35;
      v35 = v42;
      j_j___libc_free_0(v45, 48);
    }
  }
  v38 = *((_DWORD *)v35 + 10);
  if ( v38 != 4 )
  {
    if ( v38 > 4 )
    {
      if ( v38 == 5 )
        return 8;
      if ( v38 == 101 )
        return 16;
      return 15;
    }
    if ( v38 != 1 )
    {
      if ( v38 != 3 )
        return 15;
      return 2;
    }
  }
  return v38;
}
