// Function: sub_1B9C6F0
// Address: 0x1b9c6f0
//
void __fastcall sub_1B9C6F0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rax
  __int64 v10; // r13
  _QWORD *v12; // rax
  __int64 *v13; // rsi
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned int v17; // r12d
  _BYTE *v18; // r14
  _QWORD *v19; // rax
  __int64 v20; // r15
  _QWORD **v21; // r14
  __int64 v22; // rax
  unsigned __int8 v23; // al
  _QWORD *v24; // rax
  __int64 v25; // r9
  __int64 v26; // r14
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // rax
  __int64 **v30; // r12
  __int64 **i; // r13
  _QWORD *v32; // rax
  __int64 *v33; // rsi
  _QWORD *v34; // r8
  __int64 v35; // rdi
  __int64 v36; // rdx
  unsigned int v37; // r15d
  __int64 v38; // rax
  unsigned __int64 *v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rsi
  _QWORD *v43; // rax
  __int64 v44; // rax
  _BYTE *v45; // rax
  _BYTE *v46; // rdx
  _QWORD *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rsi
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rsi
  _QWORD *v54; // rax
  __int64 v55; // rsi
  _QWORD *v56; // rax
  __int64 v57; // rax
  __int64 v58; // [rsp+8h] [rbp-148h]
  __int64 v59; // [rsp+10h] [rbp-140h]
  _QWORD *v60; // [rsp+10h] [rbp-140h]
  _QWORD *v61; // [rsp+10h] [rbp-140h]
  __int64 v62; // [rsp+10h] [rbp-140h]
  __int64 *v63; // [rsp+18h] [rbp-138h]
  _QWORD *v64; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v65; // [rsp+18h] [rbp-138h]
  __int64 v66; // [rsp+20h] [rbp-130h]
  _QWORD *v67; // [rsp+28h] [rbp-128h]
  __int64 v68; // [rsp+38h] [rbp-118h]
  _QWORD *v69; // [rsp+48h] [rbp-108h] BYREF
  __int64 *v70[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v71[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v72; // [rsp+70h] [rbp-E0h]
  __int64 v73; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE *v74; // [rsp+88h] [rbp-C8h]
  _BYTE *v75; // [rsp+90h] [rbp-C0h]
  __int64 v76; // [rsp+98h] [rbp-B8h]
  int v77; // [rsp+A0h] [rbp-B0h]
  _BYTE v78[40]; // [rsp+A8h] [rbp-A8h] BYREF
  unsigned __int64 v79[5]; // [rsp+D0h] [rbp-80h] BYREF
  int v80; // [rsp+F8h] [rbp-58h]
  __int64 v81; // [rsp+100h] [rbp-50h]
  __int64 v82; // [rsp+108h] [rbp-48h]

  v74 = v78;
  v75 = v78;
  v9 = *(_QWORD *)(a1 + 456);
  v73 = 0;
  v76 = 4;
  v10 = *(_QWORD *)(v9 + 40);
  v77 = 0;
  v66 = *(_QWORD *)(v9 + 48);
  if ( v10 == v66 )
    return;
  v67 = (_QWORD *)(a1 + 296);
  do
  {
    v12 = *(_QWORD **)(a1 + 304);
    v13 = *(__int64 **)v10;
    if ( v12 )
    {
      v14 = v67;
      do
      {
        while ( 1 )
        {
          v15 = v12[2];
          v16 = v12[3];
          if ( v12[4] >= (unsigned __int64)v13 )
            break;
          v12 = (_QWORD *)v12[3];
          if ( !v16 )
            goto LABEL_8;
        }
        v14 = v12;
        v12 = (_QWORD *)v12[2];
      }
      while ( v15 );
LABEL_8:
      if ( v14 != v67 && v14[4] <= (unsigned __int64)v13 && *(_DWORD *)(a1 + 92) )
      {
        v17 = 0;
        while ( 1 )
        {
          v20 = sub_1B9C240((unsigned int *)a1, v13, v17);
          v19 = v74;
          if ( v75 == v74 )
          {
            v18 = &v74[8 * HIDWORD(v76)];
            if ( v74 == v18 )
            {
              v46 = v74;
            }
            else
            {
              do
              {
                if ( v20 == *v19 )
                  break;
                ++v19;
              }
              while ( v18 != (_BYTE *)v19 );
              v46 = &v74[8 * HIDWORD(v76)];
            }
          }
          else
          {
            v18 = &v75[8 * (unsigned int)v76];
            v19 = sub_16CC9F0((__int64)&v73, v20);
            if ( v20 == *v19 )
            {
              if ( v75 == v74 )
                v46 = &v75[8 * HIDWORD(v76)];
              else
                v46 = &v75[8 * (unsigned int)v76];
            }
            else
            {
              if ( v75 != v74 )
              {
                v19 = &v75[8 * (unsigned int)v76];
                goto LABEL_15;
              }
              v19 = &v75[8 * HIDWORD(v76)];
              v46 = v19;
            }
          }
          if ( v19 != (_QWORD *)v46 )
          {
            while ( *v19 >= 0xFFFFFFFFFFFFFFFELL )
            {
              if ( v46 == (_BYTE *)++v19 )
              {
                if ( v18 != (_BYTE *)v19 )
                  goto LABEL_16;
                goto LABEL_28;
              }
            }
          }
LABEL_15:
          if ( v18 != (_BYTE *)v19 )
            goto LABEL_16;
LABEL_28:
          if ( !*(_QWORD *)(v20 + 8)
            || *(_BYTE *)(v20 + 16) <= 0x17u
            || (v21 = *(_QWORD ***)v20,
                v63 = (__int64 *)sub_1644900(**(_QWORD ***)v20, *(_DWORD *)(v10 + 8)),
                v69 = sub_16463B0(v63, *((_DWORD *)v21 + 8)),
                v69 == v21) )
          {
LABEL_16:
            if ( *(_DWORD *)(a1 + 92) <= ++v17 )
              break;
            goto LABEL_17;
          }
          v22 = sub_16498A0(v20);
          memset(v79, 0, 24);
          v79[3] = v22;
          v79[4] = 0;
          v80 = 0;
          v81 = 0;
          v82 = 0;
          sub_17050D0((__int64 *)v79, v20);
          v70[0] = (__int64 *)&v69;
          v70[1] = (__int64 *)v79;
          v23 = *(_BYTE *)(v20 + 16);
          if ( v23 > 0x17u )
          {
            if ( (unsigned int)v23 - 35 <= 0x11 )
            {
              v72 = 257;
              v64 = sub_1B967F0(v70, *(_QWORD *)(v20 - 24));
              v24 = sub_1B967F0(v70, *(_QWORD *)(v20 - 48));
              v65 = (unsigned __int8 *)sub_1904E90(
                                         (__int64)v79,
                                         (unsigned int)*(unsigned __int8 *)(v20 + 16) - 24,
                                         (__int64)v24,
                                         (__int64)v64,
                                         v71,
                                         0,
                                         *(double *)a2.m128_u64,
                                         a3,
                                         a4);
              sub_15F2530(v65, v20, 0);
              v25 = (__int64)v65;
LABEL_34:
              v59 = v25;
              sub_164B7C0(v25, v20);
              v72 = 257;
              v26 = sub_1904B50((__int64 *)v79, v59, (__int64)v21, v71);
              sub_164D160(v20, v26, a2, a3, a4, a5, v27, v28, a8, a9);
              sub_15F20C0((_QWORD *)v20);
              sub_1412190((__int64)&v73, v20);
              v71[0] = *(_QWORD *)v10;
              *(_QWORD *)(*sub_1B99AC0((_QWORD *)(a1 + 288), (unsigned __int64 *)v71) + 8LL * v17) = v26;
              goto LABEL_35;
            }
            if ( v23 == 75 )
            {
              v72 = 257;
              v60 = sub_1B967F0(v70, *(_QWORD *)(v20 - 24));
              v45 = sub_1B967F0(v70, *(_QWORD *)(v20 - 48));
              v25 = sub_12AA0C0((__int64 *)v79, *(_WORD *)(v20 + 18) & 0x7FFF, v45, (__int64)v60, (__int64)v71);
              goto LABEL_34;
            }
            if ( v23 == 79 )
            {
              v72 = 257;
              v61 = sub_1B967F0(v70, *(_QWORD *)(v20 - 24));
              v47 = sub_1B967F0(v70, *(_QWORD *)(v20 - 48));
              v25 = sub_156B790((__int64 *)v79, *(_QWORD *)(v20 - 72), (__int64)v47, (__int64)v61, (__int64)v71, 0);
              goto LABEL_34;
            }
            if ( (unsigned int)v23 - 60 <= 0xC )
            {
              if ( v23 == 61 )
              {
                v48 = (__int64)v69;
                v72 = 257;
                if ( *(_DWORD *)(*v21[2] + 8LL) >> 8 < *(_DWORD *)(*(_QWORD *)v69[2] + 8LL) >> 8 )
                  v48 = (__int64)v21;
                v25 = sub_1904B50((__int64 *)v79, *(_QWORD *)(v20 - 24), v48, v71);
              }
              else if ( v23 == 62 )
              {
                v41 = (__int64)v69;
                v72 = 257;
                if ( *(_DWORD *)(*v21[2] + 8LL) >> 8 < *(_DWORD *)(*(_QWORD *)v69[2] + 8LL) >> 8 )
                  v41 = (__int64)v21;
                v25 = sub_1904CF0((__int64 *)v79, *(_QWORD *)(v20 - 24), v41, v71);
              }
              else
              {
                v25 = (__int64)sub_1B967F0(v70, *(_QWORD *)(v20 - 24));
              }
              goto LABEL_34;
            }
            if ( v23 == 85 )
            {
              v53 = *(_QWORD *)(**(_QWORD **)(v20 - 72) + 32LL);
              v72 = 257;
              v54 = sub_16463B0(v63, v53);
              v62 = sub_1904B50((__int64 *)v79, *(_QWORD *)(v20 - 72), (__int64)v54, v71);
              v55 = *(_QWORD *)(**(_QWORD **)(v20 - 48) + 32LL);
              v72 = 257;
              v56 = sub_16463B0(v63, v55);
              v57 = sub_1904B50((__int64 *)v79, *(_QWORD *)(v20 - 48), (__int64)v56, v71);
              v72 = 257;
              v25 = sub_14C50F0((__int64 *)v79, v62, v57, *(_QWORD *)(v20 - 24), (__int64)v71);
              goto LABEL_34;
            }
            if ( v23 != 54 && v23 != 77 )
            {
              if ( v23 == 84 )
              {
                v49 = *(_QWORD *)(**(_QWORD **)(v20 - 72) + 32LL);
                v72 = 257;
                v50 = sub_16463B0(v63, v49);
                v51 = sub_1904B50((__int64 *)v79, *(_QWORD *)(v20 - 72), (__int64)v50, v71);
                v72 = 257;
                v58 = v51;
                v52 = sub_1904B50((__int64 *)v79, *(_QWORD *)(v20 - 48), (__int64)v63, v71);
                v72 = 257;
                v25 = sub_156D8B0((__int64 *)v79, v58, v52, *(_QWORD *)(v20 - 24), (__int64)v71);
                goto LABEL_34;
              }
              if ( v23 == 83 )
              {
                v42 = *(_QWORD *)(**(_QWORD **)(v20 - 48) + 32LL);
                v72 = 257;
                v43 = sub_16463B0(v63, v42);
                v44 = sub_1904B50((__int64 *)v79, *(_QWORD *)(v20 - 48), (__int64)v43, v71);
                v72 = 257;
                v25 = sub_156D5F0((__int64 *)v79, v44, *(_QWORD *)v20, (__int64)v71);
                goto LABEL_34;
              }
            }
          }
LABEL_35:
          if ( !v79[0] )
            goto LABEL_16;
          ++v17;
          sub_161E7C0((__int64)v79, v79[0]);
          if ( *(_DWORD *)(a1 + 92) <= v17 )
            break;
LABEL_17:
          v13 = *(__int64 **)v10;
        }
      }
    }
    v10 += 16;
  }
  while ( v66 != v10 );
  v29 = *(_QWORD *)(a1 + 456);
  v30 = *(__int64 ***)(v29 + 40);
  for ( i = *(__int64 ***)(v29 + 48); i != v30; v30 += 2 )
  {
    v32 = *(_QWORD **)(a1 + 304);
    v33 = *v30;
    if ( v32 )
    {
      v34 = v67;
      do
      {
        while ( 1 )
        {
          v35 = v32[2];
          v36 = v32[3];
          if ( v32[4] >= (unsigned __int64)v33 )
            break;
          v32 = (_QWORD *)v32[3];
          if ( !v36 )
            goto LABEL_44;
        }
        v34 = v32;
        v32 = (_QWORD *)v32[2];
      }
      while ( v35 );
LABEL_44:
      if ( v34 != v67 && v34[4] <= (unsigned __int64)v33 && *(_DWORD *)(a1 + 92) )
      {
        v37 = 0;
        while ( 1 )
        {
          v38 = sub_1B9C240((unsigned int *)a1, v33, v37);
          if ( *(_BYTE *)(v38 + 16) != 61 || *(_QWORD *)(v38 + 8) )
          {
            if ( *(_DWORD *)(a1 + 92) <= ++v37 )
              break;
          }
          else
          {
            v68 = *(_QWORD *)(v38 - 24);
            sub_15F20C0((_QWORD *)v38);
            v79[0] = (unsigned __int64)*v30;
            v39 = sub_1B99AC0((_QWORD *)(a1 + 288), v79);
            v40 = v37++;
            *(_QWORD *)(*v39 + 8 * v40) = v68;
            if ( *(_DWORD *)(a1 + 92) <= v37 )
              break;
          }
          v33 = *v30;
        }
      }
    }
  }
  if ( v74 != v75 )
    _libc_free((unsigned __int64)v75);
}
