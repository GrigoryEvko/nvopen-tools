// Function: sub_1CD6ED0
// Address: 0x1cd6ed0
//
__int64 __fastcall sub_1CD6ED0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r15
  __int64 v12; // rax
  int v14; // r9d
  _QWORD **v15; // r11
  __int64 v16; // rdx
  __int64 v17; // r13
  int v18; // r8d
  unsigned int v19; // eax
  _QWORD *v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // r12
  _QWORD *v23; // r15
  __int64 v24; // rax
  char v25; // r8
  unsigned int v26; // edi
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r15
  _QWORD *v36; // rcx
  unsigned int v37; // eax
  _QWORD *v38; // rsi
  _QWORD *v39; // rdi
  _QWORD *v40; // rbx
  __int64 v41; // rax
  unsigned int v42; // edi
  __int64 v43; // rax
  __int64 ***v44; // rsi
  __int64 ***v45; // rax
  __int64 v46; // rax
  double v47; // xmm4_8
  double v48; // xmm5_8
  double v49; // xmm4_8
  double v50; // xmm5_8
  _QWORD **v52; // r12
  int v53; // esi
  int v54; // r8d
  _QWORD *v55; // [rsp+0h] [rbp-110h]
  _QWORD *v56; // [rsp+8h] [rbp-108h]
  __int64 v57; // [rsp+18h] [rbp-F8h]
  int v58; // [rsp+20h] [rbp-F0h]
  _QWORD *v59; // [rsp+28h] [rbp-E8h]
  __int64 v60; // [rsp+28h] [rbp-E8h]
  int v61; // [rsp+30h] [rbp-E0h]
  _QWORD **v62; // [rsp+30h] [rbp-E0h]
  _QWORD **v63; // [rsp+38h] [rbp-D8h]
  __int64 v64; // [rsp+38h] [rbp-D8h]
  __int64 v65; // [rsp+38h] [rbp-D8h]
  __int64 v66; // [rsp+40h] [rbp-D0h]
  unsigned int v67; // [rsp+4Ch] [rbp-C4h]
  __int64 ***v68; // [rsp+50h] [rbp-C0h]
  int v69; // [rsp+58h] [rbp-B8h]
  __int64 v72; // [rsp+70h] [rbp-A0h]
  __int64 v73; // [rsp+78h] [rbp-98h]
  __int64 v74; // [rsp+88h] [rbp-88h] BYREF
  __int64 v75; // [rsp+90h] [rbp-80h] BYREF
  _QWORD **v76; // [rsp+98h] [rbp-78h]
  __int64 v77; // [rsp+A0h] [rbp-70h]
  __int64 v78; // [rsp+A8h] [rbp-68h]
  char v79[96]; // [rsp+B0h] [rbp-60h] BYREF

  v11 = *a1;
  v12 = (a1[1] - *a1) >> 3;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  if ( !(_DWORD)v12 )
  {
    v15 = 0;
    return j___libc_free_0(v15);
  }
  v14 = 0;
  v73 = (unsigned int)v12;
  v15 = 0;
  v16 = v11;
  v17 = 0;
  v72 = 0;
  while ( 1 )
  {
    v22 = *(_QWORD *)(v16 + 8 * v17);
    v74 = v22;
    v23 = &v15[v72];
    if ( !v14 )
      goto LABEL_8;
    v18 = 1;
    v19 = (v14 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v20 = &v15[v19];
    v21 = *v20;
    if ( v22 != *v20 )
    {
      while ( v21 != -8 )
      {
        v19 = (v14 - 1) & (v18 + v19);
        v20 = &v15[v19];
        v21 = *v20;
        if ( v22 == *v20 )
          goto LABEL_4;
        ++v18;
      }
LABEL_8:
      v20 = &v15[v72];
      goto LABEL_9;
    }
LABEL_4:
    if ( v23 != v20 )
      goto LABEL_5;
LABEL_9:
    v24 = 0x17FFFFFFE8LL;
    v25 = *(_BYTE *)(v22 + 23) & 0x40;
    v26 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
    if ( v26 )
    {
      v27 = 24LL * *(unsigned int *)(v22 + 56) + 8;
      v28 = 0;
      do
      {
        v29 = v22 - 24LL * v26;
        if ( v25 )
          v29 = *(_QWORD *)(v22 - 8);
        if ( a3 == *(_QWORD *)(v29 + v27) )
        {
          v24 = 24 * v28;
          goto LABEL_16;
        }
        ++v28;
        v27 += 8;
      }
      while ( v26 != (_DWORD)v28 );
      v24 = 0x17FFFFFFE8LL;
    }
LABEL_16:
    if ( v25 )
      v30 = *(_QWORD *)(v22 - 8);
    else
      v30 = v22 - 24LL * v26;
    v31 = *(_QWORD *)(v30 + v24);
    if ( *(_BYTE *)(v31 + 16) == 56 )
    {
      v32 = *(_QWORD *)(v31 + 8);
      if ( v32 )
      {
        if ( !*(_QWORD *)(v32 + 8) )
        {
          v67 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
          v33 = *(_QWORD *)(v31 - 24LL * v67);
          if ( v22 == v33 )
          {
            if ( v33 )
            {
              v61 = v14;
              v63 = v15;
              v66 = v16;
              v68 = (__int64 ***)v31;
              v57 = sub_1455EB0(v22, a2);
              v14 = v61;
              v56 = v23;
              v34 = v66;
              v35 = 0;
              v36 = v20;
              v15 = v63;
              v69 = v61 - 1;
              while ( 1 )
              {
                if ( (_DWORD)v35 != (_DWORD)v17 )
                {
                  v40 = *(_QWORD **)(v34 + 8 * v35);
                  if ( !v14 )
                    goto LABEL_29;
                  v37 = v69 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
                  v38 = &v15[v37];
                  v39 = (_QWORD *)*v38;
                  if ( v40 != (_QWORD *)*v38 )
                  {
                    v53 = 1;
                    while ( v39 != (_QWORD *)-8LL )
                    {
                      v54 = v53 + 1;
                      v37 = (v37 + v53) & v69;
                      v38 = &v15[v37];
                      v39 = (_QWORD *)*v38;
                      if ( v40 == (_QWORD *)*v38 )
                        goto LABEL_25;
                      v53 = v54;
                    }
LABEL_29:
                    if ( *(_QWORD *)v22 == *v40 )
                    {
                      v58 = v14;
                      v59 = v36;
                      v62 = v15;
                      v64 = v34;
                      v41 = sub_1455EB0(*(_QWORD *)(v34 + 8 * v35), a3);
                      v34 = v64;
                      v15 = v62;
                      v36 = v59;
                      v14 = v58;
                      if ( *(_BYTE *)(v41 + 16) == 56 )
                      {
                        v42 = *(_DWORD *)(v41 + 20) & 0xFFFFFFF;
                        if ( v40 == *(_QWORD **)(v41 - 24LL * v42) && *(_QWORD *)(v41 - 24LL * v42) != 0 && v67 == v42 )
                        {
                          v55 = v59;
                          v60 = v64;
                          v65 = v41;
                          v43 = sub_1455EB0((__int64)v40, a2);
                          v34 = v60;
                          v15 = v62;
                          v36 = v55;
                          v14 = v58;
                          if ( v57 == v43 )
                          {
                            v44 = (__int64 ***)(v65 + 24 * (1LL - v42));
                            v45 = &v68[3 * (1LL - v67)];
                            if ( v45 == v68 )
                            {
LABEL_37:
                              v46 = sub_1599EF0(*v68);
                              sub_164D160((__int64)v68, v46, a4, a5, a6, a7, v47, v48, a10, a11);
                              sub_15F20C0(v68);
                              sub_164D160(v74, (__int64)v40, a4, a5, a6, a7, v49, v50, a10, a11);
                              sub_1CD6D80((__int64)v79, (__int64)&v75, &v74);
                              v15 = v76;
                              v72 = (unsigned int)v78;
                              v14 = v78;
                              v23 = &v76[(unsigned int)v78];
                              break;
                            }
                            while ( *v45 == *v44 )
                            {
                              v45 += 3;
                              v44 += 3;
                              if ( v45 == v68 )
                                goto LABEL_37;
                            }
                          }
                        }
                      }
                    }
                    goto LABEL_26;
                  }
LABEL_25:
                  if ( v38 == v36 )
                    goto LABEL_29;
                }
LABEL_26:
                if ( ++v35 == v73 )
                {
                  v23 = v56;
                  break;
                }
              }
            }
          }
        }
      }
    }
LABEL_5:
    if ( v73 == ++v17 )
      break;
    v16 = *a1;
  }
  if ( (_DWORD)v77 && v23 != v15 )
  {
    v52 = v15;
    while ( *v52 == (_QWORD *)-8LL || *v52 == (_QWORD *)-16LL )
    {
      if ( ++v52 == v23 )
        return j___libc_free_0(v15);
    }
    if ( v52 != v23 )
    {
LABEL_52:
      sub_15F20C0(*v52);
      while ( ++v52 != v23 )
      {
        if ( *v52 != (_QWORD *)-16LL && *v52 != (_QWORD *)-8LL )
        {
          if ( v23 != v52 )
            goto LABEL_52;
          break;
        }
      }
      v15 = v76;
    }
  }
  return j___libc_free_0(v15);
}
