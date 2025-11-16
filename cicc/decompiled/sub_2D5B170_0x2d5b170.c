// Function: sub_2D5B170
// Address: 0x2d5b170
//
bool __fastcall sub_2D5B170(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4, _QWORD *a5, __int64 *a6)
{
  bool result; // al
  unsigned __int8 *v9; // r8
  __int64 v10; // r10
  unsigned __int64 v11; // rax
  __int64 v15; // rcx
  int v16; // eax
  _BYTE *v17; // rsi
  unsigned __int8 *v18; // rax
  int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // rsi
  int v22; // edx
  unsigned int v23; // r11d
  __int64 *v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdx
  unsigned __int8 *v31; // r12
  __int64 v32; // r8
  __int64 v33; // r10
  __int64 *v34; // r9
  unsigned int v35; // ebx
  int v36; // eax
  bool v37; // al
  unsigned __int8 v38; // cl
  int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // rdx
  int v42; // eax
  __int64 v43; // rbx
  _BYTE *v44; // rax
  __int64 v45; // r10
  bool v46; // al
  bool v47; // bl
  int v48; // eax
  bool v49; // dl
  unsigned int v50; // ebx
  _BYTE *v51; // rax
  bool v52; // [rsp-89h] [rbp-89h]
  __int64 v54; // [rsp-80h] [rbp-80h]
  __int64 v55; // [rsp-80h] [rbp-80h]
  __int64 v56; // [rsp-78h] [rbp-78h]
  __int64 v57; // [rsp-78h] [rbp-78h]
  __int64 v58; // [rsp-70h] [rbp-70h]
  int v59; // [rsp-70h] [rbp-70h]
  int v60; // [rsp-70h] [rbp-70h]
  __int64 v61; // [rsp-68h] [rbp-68h]
  _BYTE *v62; // [rsp-60h] [rbp-60h]
  unsigned __int8 *v63; // [rsp-58h] [rbp-58h] BYREF
  __int64 v64; // [rsp-50h] [rbp-50h]
  char v65; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)a1 != 51 )
    return 0;
  v9 = *(unsigned __int8 **)(a1 - 64);
  if ( !v9 )
    return 0;
  v10 = *(_QWORD *)(a1 - 32);
  if ( !v10 )
    return 0;
  v11 = *v9;
  if ( (unsigned __int8)v11 <= 0x1Cu )
  {
    if ( (_BYTE)v11 != 5 )
      return 0;
    v16 = *((unsigned __int16 *)v9 + 1);
    if ( (*((_WORD *)v9 + 1) & 0xFFF7) != 0x11 && (v16 & 0xFFFD) != 0xD )
      return 0;
  }
  else
  {
    if ( (_BYTE)v11 == 84 )
    {
      v62 = 0;
      v61 = 0;
      goto LABEL_17;
    }
    if ( (unsigned __int8)v11 > 0x36u )
      return 0;
    v15 = 0x40540000000000LL;
    if ( !_bittest64(&v15, v11) )
      return 0;
    v16 = (unsigned __int8)v11 - 29;
  }
  if ( v16 != 13 )
    return 0;
  if ( (v9[1] & 2) == 0 )
    return 0;
  v17 = (_BYTE *)*((_QWORD *)v9 - 8);
  v62 = v17;
  if ( !v17 )
    return 0;
  v18 = (unsigned __int8 *)*((_QWORD *)v9 - 4);
  if ( !v18 )
    return 0;
  if ( *v17 == 84 )
  {
    v62 = (_BYTE *)*((_QWORD *)v9 - 4);
    v18 = (unsigned __int8 *)*((_QWORD *)v9 - 8);
  }
  else if ( *v18 != 84 )
  {
    return 0;
  }
  v61 = *(_QWORD *)(a1 - 64);
  v9 = v18;
LABEL_17:
  if ( (*((_DWORD *)v9 + 1) & 0x7FFFFFF) == 2 )
  {
    v19 = *(_DWORD *)(a2 + 24);
    v20 = *((_QWORD *)v9 + 5);
    v21 = *(_QWORD *)(a2 + 8);
    if ( v19 )
    {
      v22 = v19 - 1;
      v23 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( v20 != *v24 )
      {
        v42 = 1;
        while ( v25 != -4096 )
        {
          v23 = v22 & (v42 + v23);
          v59 = v42 + 1;
          v24 = (__int64 *)(v21 + 16LL * v23);
          v25 = *v24;
          if ( v20 == *v24 )
            goto LABEL_20;
          v42 = v59;
        }
        return 0;
      }
LABEL_20:
      if ( v24[1] )
      {
        v56 = v10;
        v54 = (__int64)v9;
        v58 = v24[1];
        if ( sub_D4B130(v58) )
        {
          if ( sub_D47930(v58) )
          {
            if ( (unsigned __int8)sub_B19060(v58 + 56, *(_QWORD *)(a1 + 40), v26, v27) )
            {
              if ( (unsigned __int8)sub_D48480(v58, v56, v28, v29) )
              {
                sub_2D59A60((__int64)&v63, v54, a2);
                if ( v65 )
                {
                  v31 = (unsigned __int8 *)v64;
                  v32 = v54;
                  v33 = v56;
                  v34 = a6;
                  if ( *(_BYTE *)v64 == 17 )
                  {
                    v35 = *(_DWORD *)(v64 + 32);
                    if ( v35 <= 0x40 )
                    {
                      v37 = *(_QWORD *)(v64 + 24) == 1;
                    }
                    else
                    {
                      v36 = sub_C444A0(v64 + 24);
                      v32 = v54;
                      v33 = v56;
                      v34 = a6;
                      v37 = v35 - 1 == v36;
                    }
                    if ( !v37 )
                      return 0;
                  }
                  else
                  {
                    v43 = *(_QWORD *)(v64 + 8);
                    if ( (unsigned int)*(unsigned __int8 *)(v43 + 8) - 17 > 1 )
                      return 0;
                    v44 = sub_AD7630(v64, 0, v30);
                    v45 = v56;
                    if ( v44 && *v44 == 17 )
                    {
                      v46 = sub_D94040((__int64)(v44 + 24));
                      v32 = v54;
                      v33 = v56;
                      v34 = a6;
                      v47 = v46;
                    }
                    else
                    {
                      if ( *(_BYTE *)(v43 + 8) != 17 )
                        return 0;
                      v48 = *(_DWORD *)(v43 + 32);
                      v57 = v54;
                      v55 = v45;
                      v49 = 0;
                      v50 = 0;
                      v60 = v48;
                      while ( v60 != v50 )
                      {
                        v52 = v49;
                        v51 = (_BYTE *)sub_AD69F0(v31, v50);
                        if ( !v51 )
                          return 0;
                        v49 = v52;
                        if ( *v51 != 13 )
                        {
                          if ( *v51 != 17 )
                            return 0;
                          v49 = sub_D94040((__int64)(v51 + 24));
                          if ( !v49 )
                            return 0;
                        }
                        ++v50;
                      }
                      v32 = v57;
                      v33 = v55;
                      v47 = v49;
                      v34 = a6;
                    }
                    if ( !v47 )
                      return 0;
                  }
                  v38 = *v63;
                  if ( *v63 <= 0x36u && ((0x40540000000000uLL >> v38) & 1) != 0 )
                  {
                    v39 = v38 - 29;
                    if ( v38 <= 0x1Cu )
                      v39 = *((unsigned __int16 *)v63 + 1);
                    if ( v39 == 13 )
                    {
                      result = (v63[1] & 2) != 0;
                      if ( (v63[1] & 2) != 0
                        && ((v40 = *((_QWORD *)v63 - 8)) != 0 && v40 == v32
                         || (v41 = *((_QWORD *)v63 - 4)) != 0 && v41 == v32) )
                      {
                        *a3 = v33;
                        *v34 = v32;
                        *a4 = v61;
                        *a5 = v62;
                        return result;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
