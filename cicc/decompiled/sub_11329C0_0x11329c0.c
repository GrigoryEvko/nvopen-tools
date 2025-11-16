// Function: sub_11329C0
// Address: 0x11329c0
//
_QWORD *__fastcall sub_11329C0(__int64 a1, __int64 a2, __int64 a3, const void **a4)
{
  __int64 v6; // r15
  __int64 v7; // r14
  int v8; // r12d
  __int64 v9; // r13
  _QWORD *v10; // rax
  _QWORD *v11; // r14
  unsigned int v13; // edx
  int v14; // eax
  bool v15; // al
  unsigned int v16; // eax
  __int64 *v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // r12
  _QWORD *v21; // rax
  int v22; // eax
  _BYTE *v23; // rsi
  _BYTE *v24; // rsi
  __int64 v25; // r12
  _QWORD *v26; // rax
  __int64 v27; // rbx
  _QWORD *v28; // rax
  __int64 v29; // rax
  __m128i v30; // xmm6
  __m128i v31; // xmm2
  unsigned __int64 v32; // xmm4_8
  __int64 v33; // rax
  unsigned __int32 v34; // eax
  __int64 v35; // rdx
  char v36; // cl
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  unsigned int v39; // ebx
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rbx
  _QWORD *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // r13
  _QWORD *v47; // rax
  __int64 v48; // r12
  _QWORD *v49; // rax
  __int64 v50; // r15
  _QWORD *v51; // rax
  unsigned int v52; // [rsp+Ch] [rbp-104h]
  unsigned int v53; // [rsp+Ch] [rbp-104h]
  __int16 v54; // [rsp+10h] [rbp-100h]
  unsigned int v55; // [rsp+18h] [rbp-F8h]
  unsigned int v56; // [rsp+1Ch] [rbp-F4h]
  __int64 v57; // [rsp+20h] [rbp-F0h]
  unsigned int v58; // [rsp+20h] [rbp-F0h]
  _BYTE *v60; // [rsp+28h] [rbp-E8h]
  __int64 v61; // [rsp+30h] [rbp-E0h]
  __int64 v62; // [rsp+30h] [rbp-E0h]
  __int64 v63; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v64; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v65; // [rsp+58h] [rbp-B8h]
  __int64 v66; // [rsp+60h] [rbp-B0h] BYREF
  unsigned __int32 v67; // [rsp+68h] [rbp-A8h]
  __int64 v68; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v69; // [rsp+78h] [rbp-98h]
  __int64 v70[2]; // [rsp+80h] [rbp-90h] BYREF
  __m128i v71; // [rsp+90h] [rbp-80h] BYREF
  __m128i v72; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v73; // [rsp+B0h] [rbp-60h]
  __int64 v74; // [rsp+B8h] [rbp-58h]
  __m128i v75; // [rsp+C0h] [rbp-50h]
  __int64 v76; // [rsp+D0h] [rbp-40h]

  v6 = *(_QWORD *)(a3 - 32);
  v7 = *(_QWORD *)(v6 + 8);
  v8 = *(_WORD *)(a2 + 2) & 0x3F;
  v57 = *(_QWORD *)(a3 + 8);
  v54 = *(_WORD *)(a2 + 2) & 0x3F;
  v55 = sub_BCB060(v57);
  v56 = sub_BCB060(v7);
  if ( (unsigned __int8)sub_F0C890(a1, v57, v7) )
  {
    if ( (*(_BYTE *)(a3 + 1) & 4) != 0 )
    {
      sub_C44830((__int64)&v68, a4, v56);
      v9 = sub_AD8D80(v7, (__int64)&v68);
      LOWORD(v73) = 257;
      v10 = sub_BD2C40(72, unk_3F10FD0);
      v11 = v10;
      if ( v10 )
        sub_1113300((__int64)v10, v8, v6, v9, (__int64)&v71);
      if ( v69 > 0x40 && v68 )
        j_j___libc_free_0_0(v68);
      return v11;
    }
    if ( !sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F) && (*(_BYTE *)(a3 + 1) & 2) != 0 )
    {
      sub_C449B0((__int64)&v68, a4, v56);
      v27 = sub_AD8D80(v7, (__int64)&v68);
      LOWORD(v73) = 257;
      v28 = sub_BD2C40(72, unk_3F10FD0);
      v11 = v28;
      if ( v28 )
        sub_1113300((__int64)v28, v8, v6, v27, (__int64)&v71);
LABEL_42:
      sub_969240(&v68);
      return v11;
    }
  }
  v13 = *((_DWORD *)a4 + 2);
  if ( v13 <= 0x40 )
  {
    v15 = *a4 == (const void *)1;
  }
  else
  {
    v58 = *((_DWORD *)a4 + 2);
    v14 = sub_C444A0((__int64)a4);
    v13 = v58;
    v15 = v58 - 1 == v14;
  }
  if ( v13 > 1 && v15 && v54 == 40 )
  {
    v22 = sub_BCB060(*(_QWORD *)(v6 + 8));
    if ( v22 )
    {
      v72.m128i_i64[0] = 0;
      v71.m128i_i64[0] = (__int64)&v68;
      v71.m128i_i64[1] = (unsigned int)(v22 - 1);
      v72.m128i_i64[1] = (__int64)&v68;
      v73 = v71.m128i_i64[1];
      if ( *(_BYTE *)v6 == 58 )
      {
        if ( (v23 = *(_BYTE **)(v6 - 64), *v23 == 56)
          && sub_1110F00((__int64)&v71, (__int64)v23)
          && sub_11327B0((__int64)&v72, 26, *(unsigned __int8 **)(v6 - 32))
          || (v24 = *(_BYTE **)(v6 - 32), *v24 == 56)
          && sub_1110F00((__int64)&v71, (__int64)v24)
          && sub_11327B0((__int64)&v72, 26, *(unsigned __int8 **)(v6 - 64)) )
        {
          if ( v68 )
          {
            v61 = v68;
            v25 = sub_AD64C0(*(_QWORD *)(v68 + 8), 1, 0);
            LOWORD(v73) = 257;
            v26 = sub_BD2C40(72, unk_3F10FD0);
            v11 = v26;
            if ( v26 )
              sub_1113300((__int64)v26, 40, v61, v25, (__int64)&v71);
            return v11;
          }
        }
      }
    }
  }
  v16 = (*(_WORD *)(a2 + 2) & 0x3F) - 32;
  if ( v16 > 1 )
    goto LABEL_15;
  v71.m128i_i64[0] = 0;
  v71.m128i_i64[1] = (__int64)&v63;
  if ( *(_BYTE *)v6 != 54 )
  {
LABEL_44:
    if ( v16 > 1 )
      goto LABEL_15;
    v29 = *(_QWORD *)(a3 + 16);
    if ( !v29 || *(_QWORD *)(v29 + 8) )
      goto LABEL_15;
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 && (unsigned __int8)sub_F0C790(a1, v55, v56) )
    {
      sub_F0A5D0((__int64)&v71, v56, v55);
      v60 = (_BYTE *)sub_AD8D80(v7, (__int64)&v71);
      sub_969240(v71.m128i_i64);
      LOWORD(v73) = 257;
      v62 = sub_A82350(*(unsigned int ***)(a1 + 32), (_BYTE *)v6, v60, (__int64)&v71);
      sub_C449B0((__int64)&v71, a4, v56);
      v50 = sub_AD8D80(v7, (__int64)&v71);
      sub_969240(v71.m128i_i64);
      LOWORD(v73) = 257;
      v51 = sub_BD2C40(72, unk_3F10FD0);
      v11 = v51;
      if ( v51 )
        sub_1113300((__int64)v51, v8, v62, v50, (__int64)&v71);
      return v11;
    }
    v30 = _mm_loadu_si128((const __m128i *)(a1 + 144));
    v31 = _mm_loadu_si128((const __m128i *)(a1 + 112));
    v32 = _mm_loadu_si128((const __m128i *)(a1 + 128)).m128i_u64[0];
    v33 = *(_QWORD *)(a1 + 160);
    v71 = _mm_loadu_si128((const __m128i *)(a1 + 96));
    v73 = v32;
    v76 = v33;
    v72 = v31;
    v74 = a2;
    v75 = v30;
    sub_9AC330((__int64)&v68, v6, 0, &v71);
    sub_9865C0((__int64)&v66, (__int64)&v68);
    v34 = v67;
    if ( v67 > 0x40 )
    {
      sub_C43BD0(&v66, v70);
      v34 = v67;
      v35 = v66;
      v67 = 0;
      v71.m128i_i32[2] = v34;
      v71.m128i_i64[0] = v66;
      if ( v34 > 0x40 )
      {
        v34 = sub_C44500((__int64)&v71);
LABEL_53:
        v52 = v34;
        sub_969240(v71.m128i_i64);
        sub_969240(&v66);
        if ( v56 - v55 <= v52 )
        {
          sub_C449B0((__int64)&v64, a4, v56);
          v67 = v56;
          v39 = v55 - v56;
          if ( v56 > 0x40 )
          {
            sub_C43690((__int64)&v66, 0, 0);
            v56 = v67;
            v55 = v67 + v39;
            if ( v67 == v67 + v39 )
              goto LABEL_87;
          }
          else
          {
            v66 = 0;
            if ( v55 == v56 )
            {
              v40 = 0;
              goto LABEL_59;
            }
          }
          if ( v55 <= 0x3F && v56 <= 0x40 )
          {
            v56 = v67;
            v40 = v66 | (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v39 + 64) << v55);
            goto LABEL_59;
          }
          sub_C43C90(&v66, v55, v56);
          v55 = v67;
LABEL_87:
          if ( v55 > 0x40 )
          {
            sub_C43B90(&v66, v70);
            v56 = v67;
            v41 = v66;
            goto LABEL_60;
          }
          v40 = v66;
          v56 = v55;
LABEL_59:
          v41 = v70[0] & v40;
          v66 = v41;
LABEL_60:
          v71.m128i_i64[0] = v41;
          v67 = 0;
          v71.m128i_i32[2] = v56;
          if ( v65 > 0x40 )
            sub_C43BD0(&v64, v71.m128i_i64);
          else
            v64 |= v41;
          sub_969240(v71.m128i_i64);
          sub_969240(&v66);
          v42 = sub_AD8D80(v7, (__int64)&v64);
          LOWORD(v73) = 257;
          v43 = sub_BD2C40(72, unk_3F10FD0);
          v11 = v43;
          if ( v43 )
            sub_1113300((__int64)v43, v8, v6, v42, (__int64)&v71);
          sub_969240(&v64);
          sub_969240(v70);
          goto LABEL_42;
        }
        sub_969240(v70);
        sub_969240(&v68);
LABEL_15:
        if ( !sub_9893F0(v8, (__int64)a4, &v64) )
          return 0;
        v72.m128i_i8[0] = 0;
        v71.m128i_i64[0] = (__int64)&v66;
        v71.m128i_i64[1] = (__int64)&v68;
        if ( (unsigned __int8)(*(_BYTE *)v6 - 55) > 1u )
          return 0;
        v17 = (*(_BYTE *)(v6 + 7) & 0x40) != 0
            ? *(__int64 **)(v6 - 8)
            : (__int64 *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
        if ( !*v17 )
          return 0;
        v66 = *v17;
        v18 = sub_986520(v6);
        if ( !(unsigned __int8)sub_991580((__int64)&v71.m128i_i64[1], *(_QWORD *)(v18 + 32)) )
          return 0;
        v19 = *(_QWORD **)v68;
        if ( *(_DWORD *)(v68 + 8) > 0x40u )
          v19 = (_QWORD *)*v19;
        if ( v55 != v56 - (_QWORD)v19 )
          return 0;
        if ( (_BYTE)v64 )
        {
          v20 = sub_AD6530(v7, v55);
          LOWORD(v73) = 257;
          v21 = sub_BD2C40(72, unk_3F10FD0);
          v11 = v21;
          if ( v21 )
            sub_1113300((__int64)v21, 40, v66, v20, (__int64)&v71);
        }
        else
        {
          v48 = sub_AD62B0(v7);
          LOWORD(v73) = 257;
          v49 = sub_BD2C40(72, unk_3F10FD0);
          v11 = v49;
          if ( v49 )
            sub_1113300((__int64)v49, 38, v66, v48, (__int64)&v71);
        }
        return v11;
      }
    }
    else
    {
      v71.m128i_i32[2] = v67;
      v35 = v70[0] | v66;
      v66 = v35;
      v71.m128i_i64[0] = v35;
      v67 = 0;
    }
    if ( v34 )
    {
      v36 = 64 - v34;
      v34 = 64;
      v37 = ~(v35 << v36);
      if ( v37 )
      {
        _BitScanReverse64(&v38, v37);
        v34 = v38 ^ 0x3F;
      }
    }
    goto LABEL_53;
  }
  if ( !(unsigned __int8)sub_993A50(&v71, *(_QWORD *)(v6 - 64)) )
    goto LABEL_72;
  v44 = *(_QWORD *)(v6 - 32);
  if ( !v44 )
    goto LABEL_72;
  *(_QWORD *)v71.m128i_i64[1] = v44;
  v53 = *((_DWORD *)a4 + 2);
  if ( v53 <= 0x40 )
  {
    if ( !*a4 )
      goto LABEL_69;
  }
  else if ( v53 == (unsigned int)sub_C444A0((__int64)a4) )
  {
LABEL_69:
    v45 = v55;
    LOWORD(v8) = (v54 != 32) + 35;
    goto LABEL_70;
  }
  if ( !sub_986BA0((__int64)a4) )
  {
LABEL_72:
    v16 = (*(_WORD *)(a2 + 2) & 0x3F) - 32;
    goto LABEL_44;
  }
  v45 = v53 - 1 - (unsigned int)sub_9871A0((__int64)a4);
LABEL_70:
  v46 = sub_AD64C0(v7, v45, 0);
  LOWORD(v73) = 257;
  v47 = sub_BD2C40(72, unk_3F10FD0);
  v11 = v47;
  if ( v47 )
    sub_1113300((__int64)v47, v8, v63, v46, (__int64)&v71);
  return v11;
}
