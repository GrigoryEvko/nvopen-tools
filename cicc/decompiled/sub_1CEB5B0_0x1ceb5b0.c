// Function: sub_1CEB5B0
// Address: 0x1ceb5b0
//
__int64 __fastcall sub_1CEB5B0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 j; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  bool v15; // zf
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8
  _BYTE *v21; // rsi
  __int64 v22; // r12
  const char *v23; // rax
  __int64 v24; // rdx
  const char *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  double v29; // xmm4_8
  double v30; // xmm5_8
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // rax
  _BYTE *v34; // rax
  __int64 i; // r15
  _QWORD *v36; // rax
  __int64 v37; // r13
  unsigned __int64 v38; // rax
  _BYTE *v39; // rsi
  __int64 v40; // r8
  __int64 v41; // rax
  double v42; // xmm4_8
  double v43; // xmm5_8
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // r12
  __int64 v48; // r13
  __int64 v49; // rcx
  __int64 v51; // [rsp+8h] [rbp-78h]
  __int64 v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+20h] [rbp-60h]
  char v56; // [rsp+28h] [rbp-58h]
  __int64 v57; // [rsp+30h] [rbp-50h]
  __int64 v58; // [rsp+30h] [rbp-50h]
  __int64 v59; // [rsp+38h] [rbp-48h]
  __int64 v60; // [rsp+38h] [rbp-48h]
  __int64 v61; // [rsp+40h] [rbp-40h] BYREF
  __int64 v62; // [rsp+48h] [rbp-38h]

  v12 = *(_QWORD *)(a1 + 184);
  if ( v12 != *(_QWORD *)(a1 + 192) )
    *(_QWORD *)(a1 + 192) = v12;
  v13 = *(_QWORD *)(a1 + 160);
  if ( v13 != *(_QWORD *)(a1 + 168) )
    *(_QWORD *)(a1 + 168) = v13;
  *(_QWORD *)(a1 + 208) = a2;
  v51 = a2 + 24;
  v53 = *(_QWORD *)(a2 + 32);
  if ( v53 != a2 + 24 )
  {
    v52 = a1 + 160;
    while ( 1 )
    {
      v14 = v53 - 56;
      if ( !v53 )
        v14 = 0;
      if ( !sub_15E4F60(v14) )
      {
        v54 = v14 + 72;
        v57 = *(_QWORD *)(v14 + 80);
        if ( v57 != v14 + 72 )
          break;
      }
LABEL_7:
      v53 = *(_QWORD *)(v53 + 8);
      if ( v51 == v53 )
      {
        v12 = *(_QWORD *)(a1 + 184);
        goto LABEL_33;
      }
    }
    while ( 1 )
    {
      if ( !v57 )
        BUG();
      j = *(_QWORD *)(v57 + 24);
      if ( j != v57 + 16 )
        break;
LABEL_29:
      v57 = *(_QWORD *)(v57 + 8);
      if ( v54 == v57 )
        goto LABEL_7;
    }
    while ( 1 )
    {
      if ( !j )
        BUG();
      if ( *(_BYTE *)(j - 8) != 78 || *(_BYTE *)(*(_QWORD *)(j - 48) + 16LL) )
        goto LABEL_21;
      v59 = *(_QWORD *)(j - 48);
      v22 = j - 24;
      v23 = sub_1649960(v59);
      v62 = v24;
      v61 = (__int64)v23;
      if ( sub_16D20C0(&v61, "__is_image_readonly", 0x13u, 0) != -1 )
        break;
      v25 = sub_1649960(v59);
      v62 = v26;
      v61 = (__int64)v25;
      if ( sub_16D20C0(&v61, "__is_image_readwrite", 0x14u, 0) != -1 )
      {
        sub_1CEB510(a1, j - 24);
        v15 = (unsigned __int8)sub_1C2EA30(*(_QWORD *)(v22 - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF))) == 0;
        v16 = *(_QWORD **)a2;
        if ( v15 )
        {
LABEL_28:
          v27 = sub_1643350(v16);
          v28 = sub_159C470(v27, 0, 0);
          sub_164D160(j - 24, v28, a3, a4, a5, a6, v29, v30, a9, a10);
LABEL_17:
          v61 = j - 24;
          v21 = *(_BYTE **)(a1 + 168);
          if ( v21 == *(_BYTE **)(a1 + 176) )
          {
            sub_17C2330(v52, v21, &v61);
          }
          else
          {
            if ( v21 )
            {
              *(_QWORD *)v21 = v22;
              v21 = *(_BYTE **)(a1 + 168);
            }
            *(_QWORD *)(a1 + 168) = v21 + 8;
          }
          goto LABEL_21;
        }
LABEL_16:
        v17 = sub_1643350(v16);
        v18 = sub_159C470(v17, 1, 0);
        sub_164D160(j - 24, v18, a3, a4, a5, a6, v19, v20, a9, a10);
        goto LABEL_17;
      }
LABEL_21:
      j = *(_QWORD *)(j + 8);
      if ( v57 + 16 == j )
        goto LABEL_29;
    }
    sub_1CEB510(a1, j - 24);
    v15 = (unsigned __int8)sub_1C2E970(*(_QWORD *)(j - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF) - 24)) == 0;
    v16 = *(_QWORD **)a2;
    if ( v15 )
      goto LABEL_28;
    goto LABEL_16;
  }
LABEL_33:
  v31 = (*(_QWORD *)(a1 + 192) - v12) >> 3;
  if ( (_DWORD)v31 )
  {
    v32 = 0;
    v60 = 8LL * (unsigned int)(v31 - 1);
    while ( 1 )
    {
      v33 = sub_15F2050(*(_QWORD *)(v12 + v32));
      v34 = (_BYTE *)sub_1632FA0(v33);
      j = sub_14DD210(*(__int64 **)(*(_QWORD *)(a1 + 184) + v32), v34, 0);
      if ( j )
      {
        if ( j == sub_159C4F0(**(__int64 ***)(a1 + 208)) )
        {
          v56 = 1;
        }
        else
        {
          if ( j != sub_159C540(**(__int64 ***)(a1 + 208)) )
            goto LABEL_36;
          v56 = 0;
        }
        for ( i = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 184) + v32) + 8LL); i; i = *(_QWORD *)(i + 8) )
        {
          v36 = sub_1648700(i);
          if ( *((_BYTE *)v36 + 16) == 26 && (*((_DWORD *)v36 + 5) & 0xFFFFFFF) == 3 )
          {
            v37 = *(v36 - 6);
            if ( !v56 )
              v37 = *(v36 - 3);
            v38 = sub_157EBA0(v37);
            if ( (unsigned int)sub_15F4D60(v38) <= 1 )
            {
              if ( sub_157F0B0(v37) )
              {
                for ( j = *(_QWORD *)(v37 + 48); v37 + 40 != j; j = *(_QWORD *)(j + 8) )
                {
                  v40 = j - 24;
                  if ( !j )
                    v40 = 0;
                  v61 = v40;
                  v58 = v40;
                  if ( v40 != sub_157EBA0(v37) )
                  {
                    if ( *(_QWORD *)(v58 + 8) )
                    {
                      v41 = sub_1599EF0(*(__int64 ***)v58);
                      sub_164D160(v58, v41, a3, a4, a5, a6, v42, v43, a9, a10);
                    }
                    v39 = *(_BYTE **)(a1 + 168);
                    if ( v39 == *(_BYTE **)(a1 + 176) )
                    {
                      sub_170B610(a1 + 160, v39, &v61);
                    }
                    else
                    {
                      if ( v39 )
                      {
                        *(_QWORD *)v39 = v61;
                        v39 = *(_BYTE **)(a1 + 168);
                      }
                      *(_QWORD *)(a1 + 168) = v39 + 8;
                    }
                  }
                }
              }
            }
          }
        }
      }
LABEL_36:
      if ( v60 == v32 )
        break;
      v12 = *(_QWORD *)(a1 + 184);
      v32 += 8;
    }
  }
  v44 = *(_QWORD *)(a1 + 168);
  v45 = *(_QWORD *)(a1 + 160);
  LOBYTE(j) = v44 != v45;
  v46 = (v44 - v45) >> 3;
  if ( (_DWORD)v46 )
  {
    v47 = 0;
    v48 = 8LL * (unsigned int)(v46 - 1);
    while ( 1 )
    {
      sub_15F20C0(*(_QWORD **)(v45 + v47));
      v45 = *(_QWORD *)(a1 + 160);
      if ( v48 == v47 )
        break;
      v47 += 8;
    }
    v44 = *(_QWORD *)(a1 + 168);
  }
  v49 = *(_QWORD *)(a1 + 184);
  if ( v49 != *(_QWORD *)(a1 + 192) )
    *(_QWORD *)(a1 + 192) = v49;
  if ( v45 != v44 )
    *(_QWORD *)(a1 + 168) = v45;
  return (unsigned int)j;
}
