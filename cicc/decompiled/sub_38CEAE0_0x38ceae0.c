// Function: sub_38CEAE0
// Address: 0x38ceae0
//
__int64 __fastcall sub_38CEAE0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, int a6, unsigned __int8 a7)
{
  int v7; // r10d
  unsigned __int8 v12; // cl
  __int64 v13; // rdi
  unsigned int v14; // r8d
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rdx
  _DWORD *v21; // rax
  __int64 v22; // rsi
  int v23; // edx
  __int64 (*v24)(); // rdx
  __int64 v25; // rax
  __int64 v27; // rdi
  char v28; // al
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  char v33; // dl
  int v34; // edx
  char v35; // r15
  __m128i v36; // xmm0
  int v37; // eax
  __int64 v38; // rax
  _DWORD *v39; // rdi
  __int64 (*v40)(); // rdx
  bool v41; // zf
  __int64 v42; // [rsp-10h] [rbp-A0h]
  __int64 v43; // [rsp-8h] [rbp-98h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  int v46; // [rsp+18h] [rbp-78h]
  _QWORD v47[3]; // [rsp+20h] [rbp-70h] BYREF
  int v48; // [rsp+38h] [rbp-58h]
  __m128i v49; // [rsp+40h] [rbp-50h] BYREF
  __int64 v50; // [rsp+50h] [rbp-40h]
  int v51; // [rsp+58h] [rbp-38h]

  v7 = a6;
  v12 = a7;
  switch ( *(_DWORD *)a1 )
  {
    case 0:
      v19 = *(_QWORD *)(a1 + 24);
      v45 = a5;
      memset(v47, 0, sizeof(v47));
      v48 = 0;
      v49 = 0u;
      v50 = 0;
      v51 = 0;
      if ( !(unsigned __int8)sub_38CEAE0(v19, (unsigned int)v47, a3, a4, a5, a6, a7) )
      {
        v21 = *(_DWORD **)(a1 + 24);
        if ( *v21 != 4 )
          return 0;
        v22 = *(_QWORD *)(a1 + 32);
        if ( !v22 )
          return 0;
        v23 = *(_DWORD *)(a1 + 16);
        if ( v23 == 3 )
        {
          v39 = v21 - 2;
          v40 = *(__int64 (**)())(*((_QWORD *)v21 - 1) + 40LL);
          v25 = 0;
          if ( v40 != sub_2162C20 )
            v25 = -(__int64)((unsigned __int8 (__fastcall *)(_DWORD *, __int64, __int64 (*)(), __int64, __int64))v40)(
                              v39,
                              v22,
                              v40,
                              v42,
                              v45);
        }
        else
        {
          if ( v23 != 12 )
            return 0;
          v24 = *(__int64 (**)())(*(_QWORD *)(v22 - 8) + 40LL);
          v25 = -1;
          if ( v24 != sub_2162C20 )
            v25 = -(__int64)(((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 (*)(), __int64, __int64))v24)(
                               v22 - 8,
                               v22,
                               v24,
                               v42,
                               v45)
                           ^ 1u);
        }
        goto LABEL_20;
      }
      return sub_38CEE48(v19, v43, v20, v42, v45);
    case 1:
      v25 = *(_QWORD *)(a1 + 16);
LABEL_20:
      *(_QWORD *)a2 = 0;
      v14 = 1;
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = v25;
      *(_DWORD *)(a2 + 24) = 0;
      return v14;
    case 2:
      v18 = *(_QWORD *)(a1 + 24);
      if ( (*(_BYTE *)(v18 + 9) & 0xC) != 8 )
        goto LABEL_10;
      if ( *(_WORD *)(a1 + 16) )
        goto LABEL_10;
      v27 = *(_QWORD *)(v18 + 24);
      v28 = *(_BYTE *)(v18 + 8) | 4;
      *(_BYTE *)(v18 + 8) = v28;
      if ( *(_DWORD *)v27 == 2 && *(_WORD *)(v27 + 16) == 27 )
        goto LABEL_10;
      if ( a7 )
        goto LABEL_29;
      v29 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v29 )
        goto LABEL_47;
      v46 = a5;
      v30 = (unsigned __int64)sub_38CE440(v27);
      LODWORD(a5) = v46;
      v7 = a6;
      v31 = v30;
      v12 = a7;
      v32 = v30 | *(_QWORD *)v18 & 7LL;
      *(_QWORD *)v18 = v32;
      if ( v31 )
      {
        v29 = v32 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v29 )
        {
          v29 = 0;
          if ( (*(_BYTE *)(v18 + 9) & 0xC) == 8 )
          {
            *(_BYTE *)(v18 + 8) |= 4u;
            v29 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v18 + 24));
            v12 = a7;
            v7 = a6;
            LODWORD(a5) = v46;
            *(_QWORD *)v18 = v29 | *(_QWORD *)v18 & 7LL;
          }
        }
LABEL_47:
        if ( off_4CF6DB8 != (_UNKNOWN *)v29 )
          goto LABEL_10;
      }
      v28 = *(_BYTE *)(v18 + 8);
      v27 = *(_QWORD *)(v18 + 24);
LABEL_29:
      v33 = *(_BYTE *)(a1 + 18);
      *(_BYTE *)(v18 + 8) = v28 | 4;
      v34 = v33 & 2;
      v35 = v34;
      v14 = sub_38CEAE0(v27, a2, a3, a4, a5, v7, (unsigned __int8)((v34 != 0) | v12));
      if ( !(_BYTE)v14
        || v35
        && *(_OWORD *)a2 != 0
        && (*(_QWORD *)(a2 + 16) || (LOBYTE(v14) = *(_QWORD *)(a2 + 8) == 0 || *(_QWORD *)a2 == 0, !(_BYTE)v14)) )
      {
LABEL_10:
        *(_QWORD *)a2 = a1;
        v14 = 1;
        *(_QWORD *)(a2 + 8) = 0;
        *(_QWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 24) = 0;
      }
      return v14;
    case 3:
      v13 = *(_QWORD *)(a1 + 24);
      v49 = 0u;
      v50 = 0;
      v51 = 0;
      v14 = sub_38CEAE0(v13, (unsigned int)&v49, a3, a4, a5, a6, a7);
      if ( !(_BYTE)v14 )
        return v14;
      v15 = *(_DWORD *)(a1 + 16);
      if ( v15 != 2 )
      {
        if ( v15 > 2 )
        {
          if ( v15 == 3 )
          {
            v36 = _mm_loadu_si128(&v49);
            *(_QWORD *)(a2 + 16) = v50;
            v37 = v51;
            *(__m128i *)a2 = v36;
            *(_DWORD *)(a2 + 24) = v37;
          }
          return v14;
        }
        if ( v15 )
        {
          v16 = v49.m128i_i64[0];
          if ( !v49.m128i_i64[0] || v49.m128i_i64[1] )
          {
            v17 = v50;
            *(_QWORD *)a2 = v49.m128i_i64[1];
            *(_QWORD *)(a2 + 8) = v16;
            *(_DWORD *)(a2 + 24) = 0;
            *(_QWORD *)(a2 + 16) = -v17;
            return v14;
          }
          return 0;
        }
        if ( *(_OWORD *)&v49 == 0 )
        {
          v41 = v50 == 0;
          *(_QWORD *)a2 = 0;
          *(_QWORD *)(a2 + 8) = 0;
          *(_QWORD *)(a2 + 16) = v41;
          *(_DWORD *)(a2 + 24) = 0;
          return v14;
        }
        return 0;
      }
      if ( *(_OWORD *)&v49 != 0 )
        return 0;
      v38 = v50;
      *(_QWORD *)a2 = 0;
      *(_QWORD *)(a2 + 8) = 0;
      *(_DWORD *)(a2 + 24) = 0;
      *(_QWORD *)(a2 + 16) = ~v38;
      return v14;
    case 4:
      return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)(a1 - 8) + 32LL))(
               a1 - 8,
               a2,
               a4,
               a5);
  }
}
