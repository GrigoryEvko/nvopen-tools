// Function: sub_2726B80
// Address: 0x2726b80
//
__int64 __fastcall sub_2726B80(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 *v4; // rax
  __int64 v5; // r8
  _BYTE *v6; // r9
  __int64 v7; // r14
  __int64 v8; // r13
  _BYTE *v9; // r15
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r13
  char v14; // al
  unsigned __int64 v15; // rax
  char *v16; // rax
  char v17; // al
  unsigned __int64 v19; // rax
  _QWORD *v20; // r13
  int v21; // eax
  char *v22; // rax
  char *v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int8 *v27; // rax
  unsigned __int8 v28; // al
  __int64 v29; // rax
  __int64 v30; // rsi
  int v31; // eax
  unsigned __int8 *v32; // rax
  __int64 *v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 *v41; // rax
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 v44; // [rsp+8h] [rbp-228h]
  __int64 v45; // [rsp+8h] [rbp-228h]
  char v46; // [rsp+10h] [rbp-220h]
  __int64 *v47; // [rsp+10h] [rbp-220h]
  unsigned __int8 v48; // [rsp+10h] [rbp-220h]
  unsigned __int8 v49; // [rsp+27h] [rbp-209h]
  __int64 v50; // [rsp+28h] [rbp-208h]
  _BYTE *v51; // [rsp+28h] [rbp-208h]
  __int64 *v52; // [rsp+28h] [rbp-208h]
  __int64 *v53; // [rsp+28h] [rbp-208h]
  _BYTE *v54; // [rsp+28h] [rbp-208h]
  _BYTE *v55; // [rsp+28h] [rbp-208h]
  int v56; // [rsp+34h] [rbp-1FCh] BYREF
  unsigned __int8 *v57; // [rsp+38h] [rbp-1F8h] BYREF
  __int64 *v58; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 *v59; // [rsp+48h] [rbp-1E8h] BYREF
  _BYTE *v60; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v61; // [rsp+58h] [rbp-1D8h]
  _BYTE v62[128]; // [rsp+60h] [rbp-1D0h] BYREF
  __int64 v63; // [rsp+E0h] [rbp-150h] BYREF
  char *v64; // [rsp+E8h] [rbp-148h]
  __int64 v65; // [rsp+F0h] [rbp-140h]
  int v66; // [rsp+F8h] [rbp-138h]
  char v67; // [rsp+FCh] [rbp-134h]
  char v68; // [rsp+100h] [rbp-130h] BYREF

  v49 = sub_27269F0(a1, a2, a3, &v57, &v58, &v59);
  if ( v49 )
  {
    if ( (unsigned int)*v57 - 12 <= 9 )
    {
      return 0;
    }
    else
    {
      v4 = sub_DD8400(*a1, (__int64)v57);
      v67 = 1;
      v65 = 32;
      v7 = (__int64)v4;
      v64 = &v68;
      v60 = v62;
      v61 = 0x1000000000LL;
      v66 = 0;
      v63 = 0;
      v8 = *((_QWORD *)v57 + 2);
      if ( v8 )
      {
        do
        {
          v9 = *(_BYTE **)(v8 + 24);
          if ( (_BYTE *)a2 != v9 && *v9 > 0x1Cu )
          {
            v50 = sub_B43CB0(*(_QWORD *)(v8 + 24));
            if ( v50 == sub_B43CB0(a2) )
            {
              v38 = (unsigned int)v61;
              v39 = (unsigned int)v61 + 1LL;
              if ( v39 > HIDWORD(v61) )
              {
                sub_C8D5F0((__int64)&v60, v62, v39, 8u, v5, (__int64)v6);
                v38 = (unsigned int)v61;
              }
              *(_QWORD *)&v60[8 * v38] = v9;
              LODWORD(v61) = v61 + 1;
            }
          }
          v8 = *(_QWORD *)(v8 + 8);
        }
        while ( v8 );
        v10 = v61;
        if ( (_DWORD)v61 )
        {
          while ( 1 )
          {
            v11 = (__int64)v60;
            v12 = v10;
            v13 = *(_QWORD *)&v60[8 * v10 - 8];
            LODWORD(v61) = v10 - 1;
            v14 = *(_BYTE *)v13;
            if ( *(_BYTE *)v13 == 61 )
              break;
            if ( v14 != 62 )
            {
              if ( v14 == 85 )
              {
                v26 = *(_QWORD *)(v13 - 32);
                if ( v26 )
                {
                  if ( !*(_BYTE *)v26
                    && *(_QWORD *)(v26 + 24) == *(_QWORD *)(v13 + 80)
                    && (*(_BYTE *)(v26 + 33) & 0x20) != 0 )
                  {
                    v12 = (unsigned int)(*(_DWORD *)(v26 + 36) - 238);
                    if ( (unsigned int)v12 <= 7 && ((1LL << (*(_BYTE *)(v26 + 36) + 18)) & 0xAD) != 0 )
                    {
                      if ( !(unsigned __int8)sub_98CF40(a2, v13, a1[1], 0) )
                        goto LABEL_20;
                      v52 = (__int64 *)*a1;
                      v27 = sub_BD3990(*(unsigned __int8 **)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF)), v13);
                      v28 = sub_27266D0(v7, (__int64)v58, (__int64)v59, (__int64)v27, v52);
                      v53 = (__int64 *)(v13 + 72);
                      v46 = v28;
                      if ( v28 > (unsigned __int8)sub_A74840((_QWORD *)(v13 + 72), 0) )
                      {
                        v40 = (__int64 *)sub_BD5C60(v13);
                        *(_QWORD *)(v13 + 72) = sub_A7B980(v53, v40, 1, 86);
                        v41 = (__int64 *)sub_BD5C60(v13);
                        v42 = sub_A77A40(v41, v46);
                        v56 = 0;
                        v45 = v42;
                        v43 = (__int64 *)sub_BD5C60(v13);
                        *(_QWORD *)(v13 + 72) = sub_A7B660(v53, v43, &v56, 1, v45);
                      }
                      v29 = *(_QWORD *)(v13 - 32);
                      if ( !v29 || *(_BYTE *)v29 || (v30 = *(_QWORD *)(v13 + 80), *(_QWORD *)(v29 + 24) != v30) )
                        BUG();
                      v31 = *(_DWORD *)(v29 + 36);
                      if ( v31 == 238 || (unsigned int)(v31 - 240) <= 1 )
                      {
                        v47 = (__int64 *)*a1;
                        v32 = sub_BD3990(
                                *(unsigned __int8 **)(v13 + 32 * (1LL - (*(_DWORD *)(v13 + 4) & 0x7FFFFFF))),
                                v30);
                        v48 = sub_27266D0(v7, (__int64)v58, (__int64)v59, (__int64)v32, v47);
                        if ( v48 > (unsigned __int8)sub_A74840(v53, 1) )
                        {
                          v33 = (__int64 *)sub_BD5C60(v13);
                          *(_QWORD *)(v13 + 72) = sub_A7B980(v53, v33, 2, 86);
                          v34 = (__int64 *)sub_BD5C60(v13);
                          v35 = sub_A77A40(v34, v48);
                          v56 = 1;
                          v44 = v35;
                          v36 = (__int64 *)sub_BD5C60(v13);
                          *(_QWORD *)(v13 + 72) = sub_A7B660(v53, v36, &v56, 1, v44);
                        }
                      }
                    }
                  }
                }
              }
              goto LABEL_13;
            }
            if ( !(unsigned __int8)sub_98CF40(a2, v13, a1[1], 0) )
              goto LABEL_20;
            LOWORD(v19) = sub_27266D0(v7, (__int64)v58, (__int64)v59, *(_QWORD *)(v13 - 32), (__int64 *)*a1);
            v11 = (unsigned int)v19;
            _BitScanReverse64(&v19, 1LL << (*(_WORD *)(v13 + 2) >> 1));
            LODWORD(v19) = v19 ^ 0x3F;
            v12 = (unsigned int)(63 - v19);
            if ( (unsigned __int8)v11 <= (unsigned __int8)(63 - v19) )
              goto LABEL_13;
            *(_WORD *)(v13 + 2) = (2 * (unsigned __int8)v11) | *(_WORD *)(v13 + 2) & 0xFF81;
            if ( !v67 )
              goto LABEL_30;
LABEL_14:
            v16 = v64;
            v12 = HIDWORD(v65);
            v11 = (__int64)&v64[8 * HIDWORD(v65)];
            if ( v64 == (char *)v11 )
            {
LABEL_44:
              if ( HIDWORD(v65) >= (unsigned int)v65 )
              {
LABEL_30:
                sub_C8CC70((__int64)&v63, v13, v11, v12, v5, (__int64)v6);
                v17 = *(_BYTE *)v13;
                if ( *(_BYTE *)v13 == 84 )
                {
LABEL_31:
                  v20 = *(_QWORD **)(v13 + 16);
                  if ( !v20 )
                    goto LABEL_20;
                  while ( 2 )
                  {
                    if ( *(_BYTE *)(*(_QWORD *)(*v20 + 8LL) + 8LL) == 14 )
                    {
                      v6 = (_BYTE *)v20[3];
                      if ( *v6 != 62 || (v51 = (_BYTE *)v20[3], v21 = sub_BD2910((__int64)v20), v6 = v51, v21 == 1) )
                      {
                        if ( v67 )
                        {
                          v22 = v64;
                          v23 = &v64[8 * HIDWORD(v65)];
                          if ( v64 != v23 )
                          {
                            while ( v6 != *(_BYTE **)v22 )
                            {
                              v22 += 8;
                              if ( v23 == v22 )
                                goto LABEL_41;
                            }
                            goto LABEL_33;
                          }
                        }
                        else
                        {
                          v54 = v6;
                          v37 = sub_C8CA60((__int64)&v63, (__int64)v6);
                          v6 = v54;
                          if ( v37 )
                            goto LABEL_33;
                        }
LABEL_41:
                        v24 = (unsigned int)v61;
                        v25 = (unsigned int)v61 + 1LL;
                        if ( v25 > HIDWORD(v61) )
                        {
                          v55 = v6;
                          sub_C8D5F0((__int64)&v60, v62, v25, 8u, v5, (__int64)v6);
                          v24 = (unsigned int)v61;
                          v6 = v55;
                        }
                        *(_QWORD *)&v60[8 * v24] = v6;
                        LODWORD(v61) = v61 + 1;
                      }
                    }
LABEL_33:
                    v20 = (_QWORD *)v20[1];
                    if ( !v20 )
                      goto LABEL_20;
                    continue;
                  }
                }
                goto LABEL_19;
              }
              ++HIDWORD(v65);
              *(_QWORD *)v11 = v13;
              ++v63;
            }
            else
            {
              while ( v13 != *(_QWORD *)v16 )
              {
                v16 += 8;
                if ( (char *)v11 == v16 )
                  goto LABEL_44;
              }
            }
            v17 = *(_BYTE *)v13;
            if ( *(_BYTE *)v13 == 84 )
              goto LABEL_31;
LABEL_19:
            if ( v17 == 63 )
              goto LABEL_31;
LABEL_20:
            v10 = v61;
            if ( !(_DWORD)v61 )
              goto LABEL_21;
          }
          if ( !(unsigned __int8)sub_98CF40(a2, v13, a1[1], 0) )
            goto LABEL_20;
          v11 = (unsigned __int8)sub_27266D0(v7, (__int64)v58, (__int64)v59, *(_QWORD *)(v13 - 32), (__int64 *)*a1);
          _BitScanReverse64(&v15, 1LL << (*(_WORD *)(v13 + 2) >> 1));
          LODWORD(v15) = v15 ^ 0x3F;
          v12 = (unsigned int)(63 - v15);
          if ( (unsigned __int8)v11 > (unsigned __int8)(63 - v15) )
          {
            v11 = (unsigned int)(2 * v11);
            *(_WORD *)(v13 + 2) = v11 | *(_WORD *)(v13 + 2) & 0xFF81;
          }
LABEL_13:
          if ( !v67 )
            goto LABEL_30;
          goto LABEL_14;
        }
LABEL_21:
        if ( v60 != v62 )
          _libc_free((unsigned __int64)v60);
      }
      if ( !v67 )
        _libc_free((unsigned __int64)v64);
    }
  }
  return v49;
}
