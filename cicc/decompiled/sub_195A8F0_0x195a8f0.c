// Function: sub_195A8F0
// Address: 0x195a8f0
//
__int64 __fastcall sub_195A8F0(
        __int64 a1,
        __int64 ***a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned int v13; // r13d
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // r9d
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 v21; // r8
  _BYTE *v22; // r15
  unsigned __int64 v23; // rbx
  __int64 v24; // rdi
  unsigned int v25; // r13d
  _BYTE *v26; // r8
  __int64 v27; // r13
  _QWORD *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rax
  unsigned int v35; // ebx
  int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rcx
  unsigned __int64 v41; // rdx
  __int64 v42; // rcx
  unsigned __int8 v43; // [rsp+7h] [rbp-139h]
  unsigned __int64 v44; // [rsp+10h] [rbp-130h]
  unsigned int v45; // [rsp+18h] [rbp-128h]
  _BYTE *v46; // [rsp+18h] [rbp-128h]
  unsigned int v47; // [rsp+20h] [rbp-120h]
  _BYTE *v48; // [rsp+20h] [rbp-120h]
  _BYTE *v50; // [rsp+28h] [rbp-118h]
  _BYTE *v51; // [rsp+28h] [rbp-118h]
  _BYTE *v52; // [rsp+30h] [rbp-110h] BYREF
  __int64 v53; // [rsp+38h] [rbp-108h]
  _BYTE v54[64]; // [rsp+40h] [rbp-100h] BYREF
  _BYTE *v55; // [rsp+80h] [rbp-C0h]
  __int64 v56; // [rsp+88h] [rbp-B8h]
  _BYTE v57[176]; // [rsp+90h] [rbp-B0h] BYREF

  if ( *((_BYTE *)*(a2 - 6) + 16) == 13 || *((_BYTE *)*(a2 - 3) + 16) == 13 )
    return 0;
  v11 = (__int64)a2[5];
  v12 = *(_QWORD *)(v11 + 48);
  if ( !v12 )
    BUG();
  v13 = 0;
  if ( *(_BYTE *)(v12 - 8) == 77 )
  {
    v15 = (unsigned int)*(unsigned __int8 *)(sub_157ED20((__int64)a2[5]) + 16) - 34;
    if ( (unsigned int)v15 > 0x36 || (v16 = 0x40018000000001LL, !_bittest64(&v16, v15)) )
    {
      v17 = (__int64)*(a2 - 6);
      v55 = v57;
      v56 = 0x800000000LL;
      v43 = sub_1954CE0(a1, v17, v11);
      if ( !v43 )
      {
        v13 = sub_1954CE0(a1, (__int64)*(a2 - 3), v11);
        if ( !(_BYTE)v13 )
          goto LABEL_34;
      }
      v21 = 16LL * (unsigned int)v56;
      v44 = (unsigned __int64)v55;
      v22 = &v55[v21];
      if ( v55 == &v55[v21] )
      {
        v26 = v54;
        v53 = 0x800000000LL;
        v38 = *(_QWORD *)(v11 + 48);
        v52 = v54;
        if ( v38 )
        {
          if ( (*(_DWORD *)(v38 - 4) & 0xFFFFFFF) != 0 )
          {
LABEL_31:
            v48 = v26;
            v31 = sub_195A750(a1, v11, (__int64)&v52, a3, a4, a5, a6, v19, v20, a9, a10);
            v26 = v48;
            v13 = v31;
            goto LABEL_32;
          }
          goto LABEL_48;
        }
LABEL_57:
        BUG();
      }
      v45 = 0;
      v23 = (unsigned __int64)v55;
      v47 = 0;
      while ( 1 )
      {
        v24 = *(_QWORD *)v23;
        if ( *(_BYTE *)(*(_QWORD *)v23 + 16LL) == 9 )
          goto LABEL_14;
        v25 = *(_DWORD *)(v24 + 32);
        if ( v25 <= 0x40 )
          break;
        if ( v25 == (unsigned int)sub_16A57B0(v24 + 24) )
        {
LABEL_13:
          ++v45;
LABEL_14:
          v23 += 16LL;
          if ( v22 == (_BYTE *)v23 )
            goto LABEL_19;
        }
        else
        {
LABEL_18:
          v23 += 16LL;
          ++v47;
          if ( v22 == (_BYTE *)v23 )
          {
LABEL_19:
            if ( v47 > v45 )
            {
              v34 = (__int64 *)sub_157E9C0(v11);
              v33 = sub_159C4F0(v34);
            }
            else
            {
              if ( !(v45 | v47) )
              {
                v26 = v54;
                v27 = 0;
                v52 = v54;
                v53 = 0x800000000LL;
LABEL_22:
                v28 = (_QWORD *)v44;
                v29 = 0;
                do
                {
                  if ( *v28 == v27 || *(_BYTE *)(*v28 + 16LL) == 9 )
                  {
                    if ( HIDWORD(v53) <= (unsigned int)v29 )
                    {
                      v46 = v26;
                      sub_16CD150((__int64)&v52, v26, 0, 8, (int)v26, v18);
                      v29 = (unsigned int)v53;
                      v26 = v46;
                    }
                    *(_QWORD *)&v52[8 * v29] = v28[1];
                    v29 = (unsigned int)(v53 + 1);
                    LODWORD(v53) = v53 + 1;
                  }
                  v28 += 2;
                }
                while ( v22 != (_BYTE *)v28 );
                goto LABEL_29;
              }
              v32 = (__int64 *)sub_157E9C0(v11);
              v33 = sub_159C540(v32);
            }
            v27 = v33;
            v44 = (unsigned __int64)v55;
            v22 = &v55[16 * (unsigned int)v56];
            v26 = v54;
            v52 = v54;
            v53 = 0x800000000LL;
            if ( v55 != v22 )
              goto LABEL_22;
            v29 = 0;
LABEL_29:
            v30 = *(_QWORD *)(v11 + 48);
            if ( v30 )
            {
              if ( (*(_DWORD *)(v30 - 4) & 0xFFFFFFF) != v29 )
                goto LABEL_31;
              if ( v27 )
              {
                v35 = *(_DWORD *)(v27 + 32);
                if ( v35 > 0x40 )
                {
                  v50 = v26;
                  v36 = sub_16A57B0(v27 + 24);
                  v26 = v50;
                  if ( v35 == v36 )
                    goto LABEL_44;
LABEL_50:
                  v39 = (__int64 *)&a2[3 * (v43 ^ 1u) - 6];
                  if ( *v39 )
                  {
                    v40 = v39[1];
                    v41 = v39[2] & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v41 = v40;
                    if ( v40 )
                      *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
                  }
                  *v39 = v27;
                  v42 = *(_QWORD *)(v27 + 8);
                  v39[1] = v42;
                  if ( v42 )
                    *(_QWORD *)(v42 + 16) = (unsigned __int64)(v39 + 1) | *(_QWORD *)(v42 + 16) & 3LL;
                  v39[2] = v39[2] & 3 | (v27 + 8);
                  *(_QWORD *)(v27 + 8) = v39;
                  v13 = 1;
LABEL_32:
                  if ( v52 != v26 )
                    _libc_free((unsigned __int64)v52);
LABEL_34:
                  if ( v55 != v57 )
                    _libc_free((unsigned __int64)v55);
                  return v13;
                }
                if ( *(_QWORD *)(v27 + 24) )
                  goto LABEL_50;
LABEL_44:
                v51 = v26;
                v37 = (__int64)a2[3 * v43 - 6];
LABEL_45:
                v13 = 1;
                sub_164D160((__int64)a2, v37, a3, a4, a5, a6, v19, v20, a9, a10);
                sub_15F20C0(a2);
                v26 = v51;
                goto LABEL_32;
              }
LABEL_48:
              v51 = v26;
              v37 = sub_1599EF0(*a2);
              goto LABEL_45;
            }
            goto LABEL_57;
          }
        }
      }
      if ( *(_QWORD *)(v24 + 24) )
        goto LABEL_18;
      goto LABEL_13;
    }
  }
  return v13;
}
