// Function: sub_21781F0
// Address: 0x21781f0
//
__int64 __fastcall sub_21781F0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __m128i a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  unsigned int v9; // ebx
  const __m128i *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r14
  __int64 v15; // rax
  _QWORD *v16; // rdx
  _BYTE *v17; // rax
  const void ***v19; // rax
  int v20; // edx
  __int64 v21; // r9
  __int64 *v22; // rbx
  __int64 v23; // r12
  char *v25; // rax
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rbx
  __int64 v31; // r15
  __int64 v32; // r8
  __m128i v33; // xmm2
  __m128i v34; // xmm1
  __m128i v35; // xmm0
  _OWORD *v36; // rdx
  const __m128i *v37; // r15
  __int64 v38; // rax
  unsigned int v39; // r14d
  __int64 v40; // rax
  char v41; // dl
  __int64 v42; // rax
  unsigned int v43; // eax
  __int64 v44; // r14
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r9
  __int64 v49; // r8
  unsigned __int8 v50; // dl
  unsigned int v51; // eax
  const void **v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // r9
  __int64 v55; // rax
  __int64 *v56; // rax
  unsigned __int32 v57; // eax
  __int64 v58; // r14
  __int64 v59; // rax
  int v60; // edx
  __int64 v61; // rax
  __m128i v62; // rax
  int v63; // r8d
  __int64 v64; // rdx
  __int128 v65; // [rsp-10h] [rbp-150h]
  __int128 v66; // [rsp-10h] [rbp-150h]
  __int128 v67; // [rsp-10h] [rbp-150h]
  __int64 v68; // [rsp+8h] [rbp-138h]
  __int64 v69; // [rsp+10h] [rbp-130h]
  __int64 v70; // [rsp+10h] [rbp-130h]
  __int64 v71; // [rsp+10h] [rbp-130h]
  unsigned int v72; // [rsp+10h] [rbp-130h]
  __int64 v73; // [rsp+10h] [rbp-130h]
  __int64 v74; // [rsp+18h] [rbp-128h]
  __int64 v75; // [rsp+18h] [rbp-128h]
  __int64 v76; // [rsp+18h] [rbp-128h]
  __int128 v77; // [rsp+20h] [rbp-120h] BYREF
  __int64 *v78; // [rsp+30h] [rbp-110h]
  __int64 v79; // [rsp+38h] [rbp-108h]
  __int64 v80; // [rsp+40h] [rbp-100h] BYREF
  int v81; // [rsp+48h] [rbp-F8h]
  _OWORD v82[3]; // [rsp+50h] [rbp-F0h] BYREF
  __m128i si128; // [rsp+80h] [rbp-C0h] BYREF
  _OWORD v84[11]; // [rsp+90h] [rbp-B0h] BYREF

  v12 = *(const __m128i **)(a1 + 32);
  v13 = *(_QWORD *)(a1 + 72);
  v14 = v12[2].m128i_i64[1];
  v80 = v13;
  v77 = (__int128)_mm_loadu_si128(v12);
  if ( v13 )
  {
    v69 = a2;
    sub_1623A60((__int64)&v80, v13, 2);
    a2 = v69;
  }
  v81 = *(_DWORD *)(a1 + 64);
  v15 = *(_QWORD *)(v14 + 88);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  if ( (_DWORD)v16 == 4456 )
    goto LABEL_23;
  if ( (unsigned int)v16 <= 0x1168 )
  {
    if ( (_DWORD)v16 != 4178 )
    {
      if ( (unsigned int)v16 <= 0x1051 || (_DWORD)v16 != 4424 && (_DWORD)v16 != 4434 )
        goto LABEL_20;
LABEL_23:
      v23 = (__int64)sub_2177DE0(a1, a4, a5, a6, a2, a3);
      goto LABEL_15;
    }
    v31 = *(_QWORD *)(a1 + 32);
    v32 = *(unsigned int *)(a1 + 56);
    v33 = _mm_loadu_si128((const __m128i *)v31);
    v82[0] = v33;
    v34 = _mm_loadu_si128((const __m128i *)(v31 + 40));
    v82[1] = v34;
    v35 = _mm_loadu_si128((const __m128i *)(v31 + 80));
    si128.m128i_i64[0] = (__int64)v84;
    si128.m128i_i64[1] = 0x800000003LL;
    v82[2] = v35;
    v84[0] = v33;
    v84[1] = v34;
    v84[2] = v35;
    if ( (_DWORD)v32 != 4 )
    {
      v36 = v84;
      v37 = (const __m128i *)(v31 + 160);
      v38 = 3;
      v39 = 4;
      while ( 1 )
      {
        v35 = _mm_loadu_si128(v37);
        ++v39;
        v36[v38] = v35;
        v38 = (unsigned int)++si128.m128i_i32[2];
        if ( v39 == (_DWORD)v32 )
          break;
        v37 = (const __m128i *)(*(_QWORD *)(a1 + 32) + 40LL * v39);
        if ( si128.m128i_i32[3] <= (unsigned int)v38 )
        {
          v72 = v32;
          *(_QWORD *)&v77 = &si128;
          sub_16CD150((__int64)&si128, v84, 0, 16, v32, a9);
          v38 = si128.m128i_u32[2];
          v32 = v72;
        }
        v36 = (_OWORD *)si128.m128i_i64[0];
      }
      v31 = *(_QWORD *)(a1 + 32);
    }
    v40 = *(_QWORD *)(*(_QWORD *)(v31 + 120) + 40LL) + 16LL * *(unsigned int *)(v31 + 128);
    v41 = *(_BYTE *)v40;
    v42 = *(_QWORD *)(v40 + 8);
    LOBYTE(v82[0]) = v41;
    *((_QWORD *)&v82[0] + 1) = v42;
    if ( v41 )
    {
      v50 = v41 - 14;
      if ( v50 <= 0x5Fu )
      {
        v43 = word_432BB60[v50];
LABEL_40:
        v44 = 0;
        *(_QWORD *)&v77 = v43;
        if ( v43 )
        {
          do
          {
            v45 = sub_1D38E70((__int64)a3, v44, (__int64)&v80, 0, v35, *(double *)v34.m128i_i64, v33);
            v47 = *(_QWORD *)(a1 + 32);
            v48 = v46;
            v49 = v45;
            if ( LOBYTE(v82[0]) )
            {
              switch ( LOBYTE(v82[0]) )
              {
                case 0xE:
                case 0xF:
                case 0x10:
                case 0x11:
                case 0x12:
                case 0x13:
                case 0x14:
                case 0x15:
                case 0x16:
                case 0x17:
                case 0x38:
                case 0x39:
                case 0x3A:
                case 0x3B:
                case 0x3C:
                case 0x3D:
                  LOBYTE(v51) = 2;
                  break;
                case 0x18:
                case 0x19:
                case 0x1A:
                case 0x1B:
                case 0x1C:
                case 0x1D:
                case 0x1E:
                case 0x1F:
                case 0x20:
                case 0x3E:
                case 0x3F:
                case 0x40:
                case 0x41:
                case 0x42:
                case 0x43:
                  LOBYTE(v51) = 3;
                  break;
                case 0x21:
                case 0x22:
                case 0x23:
                case 0x24:
                case 0x25:
                case 0x26:
                case 0x27:
                case 0x28:
                case 0x44:
                case 0x45:
                case 0x46:
                case 0x47:
                case 0x48:
                case 0x49:
                  LOBYTE(v51) = 4;
                  break;
                case 0x29:
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                case 0x2F:
                case 0x30:
                case 0x4A:
                case 0x4B:
                case 0x4C:
                case 0x4D:
                case 0x4E:
                case 0x4F:
                  LOBYTE(v51) = 5;
                  break;
                case 0x31:
                case 0x32:
                case 0x33:
                case 0x34:
                case 0x35:
                case 0x36:
                case 0x50:
                case 0x51:
                case 0x52:
                case 0x53:
                case 0x54:
                case 0x55:
                  LOBYTE(v51) = 6;
                  break;
                case 0x37:
                  LOBYTE(v51) = 7;
                  break;
                case 0x56:
                case 0x57:
                case 0x58:
                case 0x62:
                case 0x63:
                case 0x64:
                  LOBYTE(v51) = 8;
                  break;
                case 0x59:
                case 0x5A:
                case 0x5B:
                case 0x5C:
                case 0x5D:
                case 0x65:
                case 0x66:
                case 0x67:
                case 0x68:
                case 0x69:
                  LOBYTE(v51) = 9;
                  break;
                case 0x5E:
                case 0x5F:
                case 0x60:
                case 0x61:
                case 0x6A:
                case 0x6B:
                case 0x6C:
                case 0x6D:
                  LOBYTE(v51) = 10;
                  break;
              }
              v52 = 0;
            }
            else
            {
              v68 = *(_QWORD *)(a1 + 32);
              v71 = v45;
              v75 = v46;
              LOBYTE(v51) = sub_1F596B0((__int64)v82);
              v47 = v68;
              v49 = v71;
              v48 = v75;
              v9 = v51;
            }
            *((_QWORD *)&v67 + 1) = v48;
            LOBYTE(v9) = v51;
            *(_QWORD *)&v67 = v49;
            v32 = (__int64)sub_1D332F0(
                             a3,
                             106,
                             (__int64)&v80,
                             v9,
                             v52,
                             0,
                             *(double *)v35.m128i_i64,
                             *(double *)v34.m128i_i64,
                             v33,
                             *(_QWORD *)(v47 + 120),
                             *(_QWORD *)(v47 + 128),
                             v67);
            v54 = v53;
            v55 = si128.m128i_u32[2];
            if ( si128.m128i_i32[2] >= (unsigned __int32)si128.m128i_i32[3] )
            {
              v76 = v53;
              v73 = v32;
              sub_16CD150((__int64)&si128, v84, 0, 16, v32, v53);
              v55 = si128.m128i_u32[2];
              v32 = v73;
              v54 = v76;
            }
            v56 = (__int64 *)(si128.m128i_i64[0] + 16 * v55);
            ++v44;
            *v56 = v32;
            v56[1] = v54;
            v57 = ++si128.m128i_i32[2];
          }
          while ( (_QWORD)v77 != v44 );
        }
        else
        {
          v57 = si128.m128i_u32[2];
        }
LABEL_50:
        v58 = *(_QWORD *)(a1 + 104);
        v74 = *(_QWORD *)(a1 + 96);
        v70 = *(unsigned __int8 *)(a1 + 88);
        *(_QWORD *)&v77 = si128.m128i_i64[0];
        *((_QWORD *)&v77 + 1) = v57;
        v59 = sub_1D29190((__int64)a3, 1u, 0, si128.m128i_i64[0], v32, v70);
        v23 = sub_1D24DC0(a3, 0x2Du, (__int64)&v80, v59, v60, v58, (__int64 *)v77, *((__int64 *)&v77 + 1), v70, v74);
        if ( (_OWORD *)si128.m128i_i64[0] != v84 )
          _libc_free(si128.m128i_u64[0]);
        goto LABEL_15;
      }
    }
    else if ( sub_1F58D20((__int64)v82) )
    {
      v43 = sub_1F58D30((__int64)v82);
      goto LABEL_40;
    }
    v61 = si128.m128i_u32[2];
    if ( si128.m128i_i32[2] >= (unsigned __int32)si128.m128i_i32[3] )
    {
      sub_16CD150((__int64)&si128, v84, 0, 16, v32, a9);
      v61 = si128.m128i_u32[2];
    }
    *(__m128i *)(si128.m128i_i64[0] + 16 * v61) = _mm_loadu_si128((const __m128i *)(v31 + 120));
    v57 = ++si128.m128i_i32[2];
    goto LABEL_50;
  }
  if ( (_DWORD)v16 != 4495 )
  {
    if ( (_DWORD)v16 == 4496 )
    {
      v17 = sub_16D40F0((__int64)qword_4FBB4B0);
      if ( v17 ? *v17 : LOBYTE(qword_4FBB4B0[2]) )
      {
        v19 = (const void ***)sub_1D252B0((__int64)a3, 5, 0, 1, 0);
        v65 = v77;
        *(_QWORD *)&v77 = &v80;
        v22 = sub_1D37410(a3, 305, (__int64)&v80, v19, v20, v21, *(double *)a4.m128i_i64, a5, a6, v65);
        if ( **(_BYTE **)(a1 + 40) == 6 )
        {
          v62.m128i_i64[0] = sub_1D323C0(
                               a3,
                               (__int64)v22,
                               0,
                               v77,
                               6,
                               0,
                               *(double *)a4.m128i_i64,
                               a5,
                               *(double *)a6.m128i_i64);
          *(_QWORD *)&v84[0] = v22;
          si128 = v62;
          DWORD2(v84[0]) = 1;
          v78 = sub_1D37190((__int64)a3, (__int64)&si128, 2u, v77, v63, *(double *)a4.m128i_i64, a5, a6);
          v22 = v78;
          v79 = v64;
        }
        v23 = (__int64)v22;
        goto LABEL_15;
      }
    }
LABEL_20:
    v23 = 0;
    goto LABEL_15;
  }
  v25 = (char *)sub_16D40F0((__int64)qword_4FBB4B0);
  if ( v25 )
    v26 = *v25;
  else
    v26 = qword_4FBB4B0[2];
  if ( !v26 )
    goto LABEL_20;
  v27 = *(_QWORD *)(a1 + 32);
  v28 = *(unsigned int *)(v27 + 88);
  v29 = *(_QWORD *)(v27 + 80);
  v30 = *(_QWORD *)(v27 + 88);
  if ( *(_BYTE *)(*(_QWORD *)(v29 + 40) + 16 * v28) == 6 )
  {
    v29 = sub_1D323C0(a3, v29, v30, (__int64)&v80, 5, 0, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
    v28 = (unsigned int)v28;
  }
  *(_QWORD *)&v84[0] = v29;
  si128 = _mm_load_si128((const __m128i *)&v77);
  *((_QWORD *)&v66 + 1) = 2;
  *(_QWORD *)&v66 = &si128;
  *((_QWORD *)&v84[0] + 1) = v28 | v30 & 0xFFFFFFFF00000000LL;
  v23 = (__int64)sub_1D359D0(a3, 304, (__int64)&v80, 1, 0, 0, *(double *)a4.m128i_i64, a5, a6, v66);
LABEL_15:
  if ( v80 )
    sub_161E7C0((__int64)&v80, v80);
  return v23;
}
