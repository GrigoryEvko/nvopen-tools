// Function: sub_2CED390
// Address: 0x2ced390
//
__int64 __fastcall sub_2CED390(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  int v10; // eax
  _QWORD *v11; // rdx
  _QWORD *v12; // r8
  _QWORD *v13; // rdi
  _QWORD *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned __int64 *v17; // rdx
  unsigned __int64 *v18; // r12
  unsigned __int64 *v19; // r9
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rcx
  unsigned __int64 *v23; // r9
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rax
  unsigned __int64 *v26; // rax
  _QWORD *v27; // r8
  unsigned __int64 *v28; // r9
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  unsigned __int64 *v31; // rax
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rdi
  _QWORD *v34; // rsi
  _QWORD *v35; // rax
  unsigned __int64 v36; // rax
  bool v37; // cl
  __int64 v38; // r13
  __int64 v39; // rcx
  __int64 v40; // rax
  unsigned int v41; // r12d
  __int64 v43; // rax
  __int64 v44; // rsi
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  char v47; // di
  unsigned __int64 v48; // rdi
  __int64 v49; // r15
  __int64 v50; // r13
  unsigned __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // r8
  unsigned int v54; // edi
  __int64 *v55; // rcx
  __int64 v56; // r10
  int v57; // ecx
  int v58; // edx
  _BYTE *v59; // [rsp+8h] [rbp-58h]
  char v60; // [rsp+10h] [rbp-50h]
  _QWORD *v61; // [rsp+18h] [rbp-48h]
  _QWORD *v62; // [rsp+18h] [rbp-48h]
  _QWORD *v63; // [rsp+18h] [rbp-48h]
  unsigned __int64 v64; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v65[7]; // [rsp+28h] [rbp-38h] BYREF

  v64 = *(_QWORD *)(a2 - 32);
  v10 = sub_2CED090((__int64)a1, v64, a3, a4);
  if ( v10 == 4 )
  {
    if ( (_BYTE)qword_5014268 )
      return 1;
  }
  else if ( v10 == 16 && (unsigned __int8)sub_CE9220(a4) && unk_50142AD )
  {
    return 1;
  }
  v11 = (_QWORD *)a1[64];
  v12 = a1 + 63;
  v13 = a1 + 63;
  v14 = v11;
  if ( !v11 )
    goto LABEL_10;
  do
  {
    while ( 1 )
    {
      v15 = v14[2];
      v16 = v14[3];
      if ( v14[4] >= a2 )
        break;
      v14 = (_QWORD *)v14[3];
      if ( !v16 )
        goto LABEL_8;
    }
    v13 = v14;
    v14 = (_QWORD *)v14[2];
  }
  while ( v15 );
LABEL_8:
  if ( v12 == v13 || v13[4] > a2 )
  {
LABEL_10:
    v17 = (unsigned __int64 *)a1[40];
    v18 = a1 + 39;
    if ( v17 )
    {
      v19 = a1 + 39;
      v20 = (unsigned __int64 *)a1[40];
      do
      {
        while ( 1 )
        {
          v21 = v20[2];
          v22 = v20[3];
          if ( v20[4] >= v64 )
            break;
          v20 = (unsigned __int64 *)v20[3];
          if ( !v22 )
            goto LABEL_15;
        }
        v19 = v20;
        v20 = (unsigned __int64 *)v20[2];
      }
      while ( v21 );
LABEL_15:
      if ( v18 != v19 && v19[4] <= v64 )
      {
        v23 = a1 + 39;
        v61 = a1 + 33;
        do
        {
          while ( 1 )
          {
            v24 = v17[2];
            v25 = v17[3];
            if ( v17[4] >= v64 )
              break;
            v17 = (unsigned __int64 *)v17[3];
            if ( !v25 )
              goto LABEL_21;
          }
          v23 = v17;
          v17 = (unsigned __int64 *)v17[2];
        }
        while ( v24 );
LABEL_21:
        if ( v18 == v23 || v23[4] > v64 )
        {
          v65[0] = &v64;
          v23 = sub_2CE2AA0(a1 + 38, (__int64)v23, v65);
        }
        if ( v61 != sub_2CE0E10((__int64)(a1 + 32), v23 + 5) )
        {
          v26 = (unsigned __int64 *)a1[40];
          v27 = a1 + 32;
          if ( v26 )
          {
            v28 = a1 + 39;
            do
            {
              while ( 1 )
              {
                v29 = v26[2];
                v30 = v26[3];
                if ( v26[4] >= v64 )
                  break;
                v26 = (unsigned __int64 *)v26[3];
                if ( !v30 )
                  goto LABEL_30;
              }
              v28 = v26;
              v26 = (unsigned __int64 *)v26[2];
            }
            while ( v29 );
LABEL_30:
            if ( v18 != v28 && v28[4] <= v64 )
            {
LABEL_33:
              v32 = (_QWORD *)a1[34];
              if ( v32 )
              {
                v33 = v28[5];
                v34 = a1 + 33;
                while ( 1 )
                {
                  v36 = v32[4];
                  v37 = v36 < v33;
                  if ( v36 == v33 )
                    v37 = v32[5] < v28[6];
                  v35 = (_QWORD *)v32[3];
                  if ( !v37 )
                  {
                    v35 = (_QWORD *)v32[2];
                    v34 = v32;
                  }
                  if ( !v35 )
                    break;
                  v32 = v35;
                }
                if ( v61 != v34 )
                {
                  if ( v34[4] == v33 )
                  {
                    if ( v28[6] >= v34[5] )
                      goto LABEL_70;
                  }
                  else if ( v34[4] <= v33 )
                  {
LABEL_70:
                    v63 = v34 + 7;
                    if ( v34 + 7 == (_QWORD *)v34[9] )
                      return 0;
                    v60 = 0;
                    v41 = 0;
                    v59 = a5;
                    v49 = a3;
                    v50 = v34[9];
                    while ( 1 )
                    {
                      v51 = *(_QWORD *)(v50 + 32);
                      if ( *(_BYTE *)(*(_QWORD *)(v51 + 8) + 8LL) != 14 )
                        return 15;
                      if ( *(_BYTE *)v51 <= 0x1Cu )
                      {
                        v41 |= sub_2CED090((__int64)a1, v51, v49, a4);
                      }
                      else
                      {
                        v52 = *(unsigned int *)(v49 + 24);
                        v53 = *(_QWORD *)(v49 + 8);
                        if ( !(_DWORD)v52 )
                          goto LABEL_86;
                        v54 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                        v55 = (__int64 *)(v53 + 16LL * v54);
                        v56 = *v55;
                        if ( v51 != *v55 )
                        {
                          v57 = 1;
                          while ( v56 != -4096 )
                          {
                            v58 = v57 + 1;
                            v54 = (v52 - 1) & (v57 + v54);
                            v55 = (__int64 *)(v53 + 16LL * v54);
                            v56 = *v55;
                            if ( v51 == *v55 )
                              goto LABEL_76;
                            v57 = v58;
                          }
LABEL_86:
                          v60 = 1;
                          goto LABEL_78;
                        }
LABEL_76:
                        if ( v55 == (__int64 *)(v53 + 16 * v52) )
                          goto LABEL_86;
                        v65[0] = *(unsigned __int64 **)(v50 + 32);
                        v41 |= *(_DWORD *)sub_2791170(v49, (__int64 *)v65);
                      }
LABEL_78:
                      v50 = sub_220EF30(v50);
                      if ( v63 == (_QWORD *)v50 )
                      {
                        if ( v41 != 15 && v60 )
                          *v59 = 1;
                        return v41;
                      }
                    }
                  }
                }
              }
              else
              {
                v34 = a1 + 33;
              }
              v65[0] = v28 + 5;
              v34 = (_QWORD *)sub_2CE4CC0(v27, v34, (const __m128i **)v65);
              goto LABEL_70;
            }
          }
          else
          {
            v28 = a1 + 39;
          }
          v65[0] = &v64;
          v31 = sub_2CE2AA0(a1 + 38, (__int64)v28, v65);
          v27 = a1 + 32;
          v28 = v31;
          goto LABEL_33;
        }
      }
    }
    return 15;
  }
  v38 = (__int64)(a1 + 63);
  do
  {
    while ( 1 )
    {
      v39 = v11[2];
      v40 = v11[3];
      if ( v11[4] >= a2 )
        break;
      v11 = (_QWORD *)v11[3];
      if ( !v40 )
        goto LABEL_45;
    }
    v38 = (__int64)v11;
    v11 = (_QWORD *)v11[2];
  }
  while ( v39 );
LABEL_45:
  if ( v12 == (_QWORD *)v38 || *(_QWORD *)(v38 + 32) > a2 )
  {
    v62 = a1 + 63;
    v43 = sub_22077B0(0x30u);
    v44 = v38;
    *(_QWORD *)(v43 + 32) = a2;
    v38 = v43;
    *(_DWORD *)(v43 + 40) = 0;
    v45 = sub_2CE4FD0(a1 + 62, v44, (unsigned __int64 *)(v43 + 32));
    if ( v46 )
    {
      v47 = v62 == v46 || v45 || v46[4] > a2;
      sub_220F040(v47, v38, v46, v62);
      ++a1[67];
    }
    else
    {
      v48 = v38;
      v38 = (__int64)v45;
      j_j___libc_free_0(v48);
    }
  }
  v41 = *(_DWORD *)(v38 + 40);
  if ( v41 > 6 )
    return (unsigned int)(v41 == 101) + 15;
  if ( !v41 )
    return 15;
  switch ( v41 )
  {
    case 1u:
    case 4u:
      return v41;
    case 3u:
      v41 = 2;
      break;
    case 5u:
      v41 = 8;
      break;
    case 6u:
      v41 = 32;
      break;
    default:
      return 15;
  }
  return v41;
}
