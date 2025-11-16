// Function: sub_1CFB870
// Address: 0x1cfb870
//
__int64 __fastcall sub_1CFB870(
        __m128 a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        __m128 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 *v17; // r15
  __int64 v18; // rax
  unsigned int v19; // r15d
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r14
  __int64 v22; // rbx
  _QWORD *v23; // r13
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // r14
  __int64 *v29; // rbx
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  _QWORD *v34; // rax
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // r14
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 *v41; // [rsp+8h] [rbp-218h]
  __int64 v42; // [rsp+18h] [rbp-208h]
  __int64 *v43; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v44; // [rsp+48h] [rbp-1D8h]
  __int64 *v45; // [rsp+50h] [rbp-1D0h] BYREF
  __int64 *v46; // [rsp+58h] [rbp-1C8h]
  __int64 *v47; // [rsp+60h] [rbp-1C0h]
  __int64 v48[4]; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v49[2]; // [rsp+90h] [rbp-190h] BYREF
  __int16 v50; // [rsp+A0h] [rbp-180h]
  _BYTE *v51; // [rsp+B0h] [rbp-170h] BYREF
  __int64 v52; // [rsp+B8h] [rbp-168h]
  _BYTE v53[256]; // [rsp+C0h] [rbp-160h] BYREF
  __int64 v54; // [rsp+1C0h] [rbp-60h]
  __int64 v55; // [rsp+1C8h] [rbp-58h]
  __int64 v56; // [rsp+1D0h] [rbp-50h]
  __int64 v57; // [rsp+1D8h] [rbp-48h]
  __int64 v58; // [rsp+1E0h] [rbp-40h]
  __int64 v59; // [rsp+1E8h] [rbp-38h]

  v10 = a10 + 72;
  v11 = *(_QWORD *)(a10 + 80);
  v45 = 0;
  v46 = 0;
  v47 = 0;
  if ( v11 == a10 + 72 )
    return 1;
  do
  {
    if ( !v11 )
      BUG();
    v12 = *(_QWORD *)(v11 + 24);
    v13 = v11 + 16;
    if ( v12 != v11 + 16 )
    {
      while ( 1 )
      {
        if ( !v12 )
          BUG();
        if ( *(_BYTE *)(v12 - 8) != 78 )
          goto LABEL_5;
        v14 = *(_QWORD *)(v12 - 48);
        if ( *(_BYTE *)(v14 + 16) || (*(_BYTE *)(v14 + 33) & 0x20) == 0 )
          goto LABEL_5;
        v51 = (_BYTE *)(v12 - 24);
        v15 = *(_QWORD *)(v12 - 48);
        if ( *(_BYTE *)(v15 + 16) )
          BUG();
        if ( *(_DWORD *)(v15 + 36) != 3660 )
          goto LABEL_5;
        v16 = v46;
        if ( v46 == v47 )
        {
          sub_1C991F0((__int64)&v45, v46, &v51);
LABEL_5:
          v12 = *(_QWORD *)(v12 + 8);
          if ( v13 == v12 )
            break;
        }
        else
        {
          if ( v46 )
          {
            *v46 = v12 - 24;
            v16 = v46;
          }
          v46 = v16 + 1;
          v12 = *(_QWORD *)(v12 + 8);
          if ( v13 == v12 )
            break;
        }
      }
    }
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v10 != v11 );
  v17 = v45;
  v41 = v46;
  if ( v46 != v45 )
  {
    v43 = v45;
    do
    {
      v18 = *v43;
      v51 = v53;
      v52 = 0x2000000000LL;
      v42 = v18;
      v54 = 0;
      v55 = 0;
      v56 = 0;
      v57 = 0;
      v58 = v18;
      v59 = 0;
      sub_1CFB5A0((__int64)&v51, v18);
      v19 = v52;
      v20 = (unsigned __int64)v51;
LABEL_20:
      v44 = v20;
      v21 = v20 + 8LL * v19;
      while ( v19 )
      {
        v22 = *(_QWORD *)(v21 - 8);
        --v19;
        v21 -= 8LL;
        LODWORD(v52) = v19;
        v59 = v22;
        v23 = sub_1648700(v22);
        if ( *((_BYTE *)v23 + 16) > 0x17u )
        {
          v20 = v44;
          switch ( *((_BYTE *)v23 + 16) )
          {
            case 0x18:
            case 0x19:
            case 0x1A:
            case 0x1B:
            case 0x1C:
            case 0x1D:
            case 0x1E:
            case 0x1F:
            case 0x20:
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x28:
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x39:
            case 0x3A:
            case 0x3C:
            case 0x3D:
            case 0x3E:
            case 0x3F:
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x49:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x55:
            case 0x56:
            case 0x57:
            case 0x58:
              goto LABEL_20;
            case 0x36:
              if ( sub_15F32D0((__int64)v23)
                || (*((_BYTE *)v23 + 18) & 1) != 0
                || *(_DWORD *)(*(_QWORD *)*(v23 - 3) + 8LL) > 0x1FFu )
              {
                goto LABEL_29;
              }
              v27 = (__int64 *)sub_15F2050((__int64)v23);
              v28 = (__int64 *)*(v23 - 3);
              v29 = v27;
              v49[0] = *v23;
              v49[1] = *v28;
              v30 = sub_15E26F0(v27, 4043, v49, 2);
              v31 = sub_1643360((_QWORD *)*v29);
              v32 = sub_159C470(v31, 0, 0);
              v48[1] = (__int64)v28;
              v50 = 257;
              v48[0] = v32;
              v48[2] = *(_QWORD *)(v58 + 24 * (1LL - (*(_DWORD *)(v58 + 20) & 0xFFFFFFF)));
              v33 = *(_QWORD *)(*(_QWORD *)v30 + 24LL);
              v34 = sub_1648AB0(72, 4u, 0);
              v37 = (__int64)v34;
              if ( v34 )
              {
                sub_15F1EA0((__int64)v34, **(_QWORD **)(v33 + 16), 54, (__int64)(v34 - 12), 4, (__int64)v23);
                *(_QWORD *)(v37 + 56) = 0;
                sub_15F5B40(v37, v33, v30, v48, 3, (__int64)v49, 0, 0);
              }
              sub_164D160((__int64)v23, v37, a1, a2, a3, a4, v35, v36, a7, a8);
              sub_15F20C0(v23);
              v20 = (unsigned __int64)v51;
              v19 = v52;
              break;
            case 0x37:
              sub_1CFA790((__int64)&v51, (__int64)v23, a1, a2, a3, a4, v24, v25, a7, a8);
              v20 = (unsigned __int64)v51;
              v19 = v52;
              break;
            case 0x38:
              if ( (unsigned int)sub_1648720(v22) )
                goto LABEL_29;
              goto LABEL_25;
            case 0x3B:
              sub_1CFAC70((__int64)&v51, (__int64)v23, a1, a2, a3, a4, v24, v25, a7, a8);
              v20 = (unsigned __int64)v51;
              v19 = v52;
              break;
            case 0x47:
            case 0x48:
            case 0x4D:
            case 0x4F:
LABEL_25:
              sub_1CFB5A0((__int64)&v51, (__int64)v23);
              v20 = (unsigned __int64)v51;
              v19 = v52;
              break;
            case 0x4E:
              v26 = *(v23 - 3);
              if ( !*(_BYTE *)(v26 + 16) && *(_DWORD *)(v26 + 36) )
              {
                sub_1CFAE70((__int64)&v51, (__int64)v23, a1, a2, a3, a4, v24, v25, a7, a8);
LABEL_29:
                v20 = (unsigned __int64)v51;
                v19 = v52;
              }
              break;
          }
          goto LABEL_20;
        }
      }
      j___libc_free_0(v55);
      if ( v51 != v53 )
        _libc_free((unsigned __int64)v51);
      sub_164D160(v42, *(_QWORD *)(v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF)), a1, a2, a3, a4, v38, v39, a7, a8);
      sub_15F20C0((_QWORD *)v42);
      ++v43;
    }
    while ( v41 != v43 );
    v17 = v45;
  }
  if ( v17 )
    j_j___libc_free_0(v17, (char *)v47 - (char *)v17);
  return 1;
}
