// Function: sub_2586040
// Address: 0x2586040
//
__int64 __fastcall sub_2586040(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // r12
  int v12; // r14d
  __int64 v13; // r12
  __int64 i; // rax
  int v15; // edx
  unsigned int v16; // eax
  __int64 v17; // r8
  unsigned __int64 v18; // r10
  int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int8 v35; // cl
  unsigned __int64 v36; // rax
  __int64 v37; // rdi
  unsigned __int64 v38; // rax
  __int64 v39; // rbx
  unsigned __int8 *v40; // rax
  __int64 v41; // rax
  int v42; // r9d
  int v43; // r8d
  unsigned int v44; // eax
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int16 v52; // cx
  unsigned __int64 v53; // rax
  unsigned int v54; // eax
  unsigned int v55; // edx
  unsigned int v56; // ecx
  int v57; // edx
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdx
  __int64 v60; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v61; // [rsp+8h] [rbp-D8h]
  __int64 v62; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v63; // [rsp+10h] [rbp-D0h]
  unsigned int v64; // [rsp+10h] [rbp-D0h]
  __int64 v65; // [rsp+10h] [rbp-D0h]
  unsigned int v68; // [rsp+2Ch] [rbp-B4h]
  void *v70; // [rsp+38h] [rbp-A8h]
  _QWORD *v71; // [rsp+40h] [rbp-A0h]
  __int64 v72; // [rsp+48h] [rbp-98h]
  __int64 *v73; // [rsp+50h] [rbp-90h]
  __int64 v74; // [rsp+50h] [rbp-90h]
  unsigned __int8 *v75; // [rsp+50h] [rbp-90h]
  __int64 v76; // [rsp+58h] [rbp-88h]
  __int64 v77; // [rsp+68h] [rbp-78h] BYREF
  __int64 v78; // [rsp+70h] [rbp-70h] BYREF
  void *v79; // [rsp+78h] [rbp-68h]
  __int64 v80; // [rsp+80h] [rbp-60h]
  __int64 v81; // [rsp+88h] [rbp-58h]
  __int64 v82; // [rsp+90h] [rbp-50h]
  __int64 v83; // [rsp+98h] [rbp-48h]
  __int64 v84; // [rsp+A0h] [rbp-40h]
  __int64 v85; // [rsp+A8h] [rbp-38h]

  v8 = sub_2568740(a3, a4);
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  sub_C7D6A0(0, 0, 8);
  v9 = *(unsigned int *)(v8 + 24);
  LODWORD(v81) = v9;
  if ( (_DWORD)v9 )
  {
    v79 = (void *)sub_C7D670(8 * v9, 8);
    v80 = *(_QWORD *)(v8 + 16);
    memcpy(v79, *(const void **)(v8 + 8), 8LL * (unsigned int)v81);
  }
  else
  {
    v79 = 0;
    v80 = 0;
  }
  v82 = *(_QWORD *)(v8 + 32);
  v83 = *(_QWORD *)(v8 + 40);
  v84 = *(_QWORD *)(v8 + 48);
  v85 = *(_QWORD *)(v8 + 56);
  sub_C7D6A0(0, 0, 8);
  v10 = *(_DWORD *)(a3 + 224);
  v70 = 0;
  v68 = v10;
  if ( v10 )
  {
    v11 = 8LL * v10;
    v70 = (void *)sub_C7D670(v11, 8);
    memcpy(v70, *(const void **)(a3 + 208), v11);
  }
  v12 = 0;
  v13 = *(_QWORD *)(a3 + 240);
  v76 = *(_QWORD *)(a3 + 248);
  v71 = (_QWORD *)(a1 + 72);
  v72 = *(_QWORD *)(a3 + 256);
  for ( i = 0; *(_DWORD *)(a5 + 40) > (unsigned int)i; v12 = i )
  {
    while ( 1 )
    {
      v20 = *(__int64 **)(*(_QWORD *)(a5 + 32) + 8 * i);
      v21 = v20[3];
      v73 = v20;
      if ( *(_BYTE *)v21 <= 0x1Cu )
        goto LABEL_11;
      v22 = v21 & 0xFFFFFFFFFFFFFFFBLL;
      v23 = v21 | 4;
      if ( (_DWORD)v81 )
        break;
LABEL_14:
      v24 = v83;
      while ( v13 != v24 || v76 != v84 || v72 != v85 )
      {
        v24 = sub_3106C80(&v78);
        v83 = v24;
        if ( v21 == v24 )
          goto LABEL_8;
      }
      i = (unsigned int)(v12 + 1);
      v12 = i;
      if ( *(_DWORD *)(a5 + 40) <= (unsigned int)i )
        goto LABEL_20;
    }
    v15 = v81 - 1;
    v16 = (v81 - 1) & (v23 ^ (v23 >> 9));
    v17 = *((_QWORD *)v79 + v16);
    if ( v23 != v17 )
    {
      v42 = 1;
      while ( v17 != -4 )
      {
        v16 = v15 & (v42 + v16);
        v17 = *((_QWORD *)v79 + v16);
        if ( v23 == v17 )
          goto LABEL_8;
        ++v42;
      }
      v43 = 1;
      v44 = v15 & (v22 ^ (v22 >> 9));
      v45 = *((_QWORD *)v79 + v44);
      if ( v22 != v45 )
      {
        while ( v45 != -4 )
        {
          v44 = v15 & (v43 + v44);
          v45 = *((_QWORD *)v79 + v44);
          if ( v22 == v45 )
            goto LABEL_8;
          ++v43;
        }
        goto LABEL_14;
      }
    }
LABEL_8:
    v18 = sub_250D070(v71);
    v19 = *(unsigned __int8 *)v21;
    if ( (unsigned int)(v19 - 67) <= 0xC )
    {
      if ( (_BYTE)v19 != 76 )
      {
LABEL_22:
        v26 = *(_QWORD *)(v21 + 16);
        if ( v26 )
        {
          v74 = v13;
          v27 = v26;
          do
          {
            v77 = v27;
            sub_25789E0(a5, &v77);
            v27 = *(_QWORD *)(v27 + 8);
          }
          while ( v27 );
          v13 = v74;
        }
      }
    }
    else
    {
      if ( (_BYTE)v19 != 63 )
      {
        if ( (unsigned __int8)(v19 - 34) > 0x33u )
          goto LABEL_11;
        v28 = 0x8000000000041LL;
        if ( _bittest64(&v28, (unsigned int)(v19 - 34)) )
        {
          if ( *(char *)(v21 + 7) < 0 )
          {
            v61 = v18;
            v29 = sub_BD2BC0(v21);
            v18 = v61;
            v62 = v30 + v29;
            if ( *(char *)(v21 + 7) < 0 )
            {
              v31 = sub_BD2BC0(v21);
              v18 = v61;
              if ( (unsigned int)((v62 - v31) >> 4) )
              {
                v64 = *(_DWORD *)(v21 + 4) & 0x7FFFFFF;
                if ( *(char *)(v21 + 7) < 0 )
                {
                  v46 = sub_BD2BC0(v21);
                  v18 = v61;
                  if ( *(char *)(v21 + 7) >= 0 )
                  {
                    if ( (unsigned int)((v46 + v47) >> 4) )
LABEL_71:
                      BUG();
                  }
                  else
                  {
                    v60 = v46 + v47;
                    v48 = sub_BD2BC0(v21);
                    v18 = v61;
                    if ( (unsigned int)((v60 - v48) >> 4) )
                    {
                      if ( *(char *)(v21 + 7) >= 0 )
                        goto LABEL_71;
                      v65 = ((__int64)v73 - (v21 - 32LL * v64)) >> 5;
                      v49 = sub_BD2BC0(v21);
                      v18 = v61;
                      if ( *(_DWORD *)(v49 + 8) <= (unsigned int)v65 )
                      {
                        if ( *(char *)(v21 + 7) >= 0 )
                          BUG();
                        v50 = sub_BD2BC0(v21);
                        if ( *(_DWORD *)(v50 + v51 - 4) > (unsigned int)v65 )
                          goto LABEL_11;
                        v18 = v61;
                      }
                    }
                  }
                }
              }
            }
          }
          if ( v73 == (__int64 *)(v21 - 32) )
            goto LABEL_11;
          v63 = v18;
          v32 = sub_254C9B0(v21, ((__int64)v73 - (v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF))) >> 5);
          v34 = sub_2584D90(a2, v32, v33, a1, 2, 0, 1);
          v35 = 0;
          v18 = v63;
          if ( v34 )
          {
            v36 = *(_QWORD *)(v34 + 96);
            v35 = -1;
            if ( v36 )
            {
              _BitScanReverse64(&v36, v36);
              v35 = 63 - (v36 ^ 0x3F);
            }
            LOBYTE(v34) = 1;
          }
          LOBYTE(v19) = *(_BYTE *)v21;
        }
        else
        {
          LOBYTE(v34) = 0;
          v35 = 0;
        }
        v37 = *v73;
        if ( (_BYTE)v19 == 62 || (_BYTE)v19 == 61 )
        {
          if ( v37 != *(_QWORD *)(v21 - 32) )
            goto LABEL_39;
          v52 = *(_WORD *)(v21 + 2) >> 1;
        }
        else
        {
          if ( (_BYTE)v19 == 66 )
          {
            if ( v37 == *(_QWORD *)(v21 - 64) )
            {
              v52 = *(_WORD *)(v21 + 2) >> 9;
              goto LABEL_69;
            }
LABEL_39:
            if ( !(_BYTE)v34 )
              goto LABEL_11;
LABEL_40:
            v75 = (unsigned __int8 *)v18;
            v38 = *(_QWORD *)(a1 + 96);
            if ( v38 )
            {
              _BitScanReverse64(&v38, v38);
              if ( v35 > (unsigned __int8)(63 - (v38 ^ 0x3F)) )
              {
                v39 = 1LL << v35;
                v40 = sub_25536C0(v37, &v77, *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL), 1);
                if ( v75 == v40 && v40 )
                {
                  v54 = v39;
                  v55 = abs32(v77);
                  if ( (_DWORD)v77 )
                  {
                    v56 = v55 & (v39 - 1);
                    if ( (_DWORD)v39 )
                    {
                      while ( v56 )
                      {
                        v57 = v54 % v56;
                        v54 = v56;
                        v56 = v57;
                      }
                      v55 = v54;
                    }
LABEL_78:
                    _BitScanReverse(&v55, v55);
                    v58 = 0x80000000 >> (v55 ^ 0x1F);
                    v59 = v58;
                    if ( *(_QWORD *)(a6 + 16) >= v58 )
                      v59 = *(_QWORD *)(a6 + 16);
                    if ( *(_QWORD *)(a6 + 8) >= v58 )
                      v58 = *(_QWORD *)(a6 + 8);
                    *(_QWORD *)(a6 + 16) = v59;
                    *(_QWORD *)(a6 + 8) = v58;
                    goto LABEL_11;
                  }
                  v55 = v39;
                  if ( (_DWORD)v39 )
                    goto LABEL_78;
                }
                else
                {
                  v39 = (unsigned int)v39;
                  v41 = v39;
                  if ( *(_QWORD *)(a6 + 16) >= (unsigned __int64)(unsigned int)v39 )
                    v41 = *(_QWORD *)(a6 + 16);
                  if ( *(_QWORD *)(a6 + 8) >= (unsigned __int64)(unsigned int)v39 )
                    v39 = *(_QWORD *)(a6 + 8);
                  *(_QWORD *)(a6 + 16) = v41;
                  *(_QWORD *)(a6 + 8) = v39;
                }
              }
            }
            goto LABEL_11;
          }
          if ( (_BYTE)v19 != 65 || v37 != *(_QWORD *)(v21 - 96) )
            goto LABEL_39;
          LOBYTE(v52) = *(_BYTE *)(v21 + 3);
        }
LABEL_69:
        _BitScanReverse64(&v53, 1LL << v52);
        v35 = 63 - (v53 ^ 0x3F);
        goto LABEL_40;
      }
      if ( (unsigned __int8)sub_B4DD90(v21) )
        goto LABEL_22;
    }
LABEL_11:
    i = (unsigned int)(v12 + 1);
  }
LABEL_20:
  sub_C7D6A0((__int64)v70, 8LL * v68, 8);
  return sub_C7D6A0((__int64)v79, 8LL * (unsigned int)v81, 8);
}
