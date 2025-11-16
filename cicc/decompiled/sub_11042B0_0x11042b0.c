// Function: sub_11042B0
// Address: 0x11042b0
//
_QWORD *__fastcall sub_11042B0(const __m128i *a1, __int64 a2)
{
  unsigned __int8 *v4; // rbx
  __int64 **v5; // r14
  __int64 v6; // r15
  int v7; // ecx
  __int64 v8; // rax
  int v10; // edx
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // r10
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r13
  __int64 v18; // r15
  __int64 v19; // rsi
  char v20; // al
  unsigned __int8 *v21; // r10
  char v22; // bl
  bool v23; // bl
  _BYTE *v24; // rdi
  _BYTE *v25; // r13
  const char *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdi
  char v29; // dl
  __int64 v30; // rax
  __int64 v31; // r11
  __int64 v32; // r13
  __int64 v33; // r14
  __int64 v34; // r15
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rbx
  _BYTE *v38; // rax
  char v39; // al
  __int64 v40; // r10
  unsigned int **v41; // rdi
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // r14
  __int64 v45; // r15
  __int64 v46; // rdx
  unsigned int v47; // esi
  unsigned int v48; // ecx
  _BYTE *v49; // rax
  unsigned int v50; // ecx
  char v51; // al
  const char *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r11
  unsigned int **v55; // rdi
  __int64 v56; // rax
  int v57; // [rsp+Ch] [rbp-A4h]
  int v58; // [rsp+10h] [rbp-A0h]
  __int64 v59; // [rsp+10h] [rbp-A0h]
  _BYTE *v60; // [rsp+10h] [rbp-A0h]
  __int64 v61; // [rsp+10h] [rbp-A0h]
  unsigned int v62; // [rsp+10h] [rbp-A0h]
  unsigned int v63; // [rsp+18h] [rbp-98h]
  __int64 v64; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v65; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v66; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v67; // [rsp+18h] [rbp-98h]
  unsigned int **v68; // [rsp+18h] [rbp-98h]
  __int64 v69; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v70; // [rsp+18h] [rbp-98h]
  __int64 v71; // [rsp+18h] [rbp-98h]
  __int64 v72; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v73; // [rsp+18h] [rbp-98h]
  __int64 v74; // [rsp+18h] [rbp-98h]
  _BYTE v75[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v76; // [rsp+40h] [rbp-70h]
  const char *v77; // [rsp+50h] [rbp-60h] BYREF
  __int64 v78; // [rsp+58h] [rbp-58h]
  __int16 v79; // [rsp+70h] [rbp-40h]

  v4 = *(unsigned __int8 **)(a2 - 32);
  v5 = *(__int64 ***)(a2 + 8);
  v6 = *((_QWORD *)v4 + 1);
  v63 = sub_BCB060(v6);
  v7 = sub_BCB060((__int64)v5);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 > 1 )
  {
    v58 = v7;
    if ( !(unsigned __int8)sub_F0C890((__int64)a1, v6, (__int64)v5) )
      return 0;
    v4 = *(unsigned __int8 **)(a2 - 32);
    v7 = v58;
  }
  v8 = *((_QWORD *)v4 + 2);
  if ( !v8 )
    return 0;
  if ( *(_QWORD *)(v8 + 8) )
    return 0;
  v10 = *v4;
  if ( (unsigned __int8)v10 <= 0x1Cu )
    return 0;
  v11 = (unsigned int)(v10 - 42);
  v12 = *((_QWORD *)v4 - 8);
  v13 = *((_QWORD *)v4 - 4);
  switch ( (int)v11 )
  {
    case 0:
    case 2:
    case 4:
    case 15:
    case 16:
    case 17:
      v14 = *(_BYTE *)v12;
      if ( *(_BYTE *)v12 <= 0x15u )
      {
        v59 = *((_QWORD *)v4 - 4);
        v15 = sub_AD4C30(*((_QWORD *)v4 - 8), v5, 0);
        v16 = a1[2].m128i_i64[0];
        v64 = v15;
        v76 = 257;
        if ( v5 == *(__int64 ***)(v59 + 8) )
        {
          v17 = v59;
        }
        else
        {
          v17 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 **))(**(_QWORD **)(v16 + 80) + 120LL))(
                  *(_QWORD *)(v16 + 80),
                  38,
                  v59,
                  v5);
          if ( !v17 )
          {
            v79 = 257;
            v17 = sub_B51D30(38, v59, (__int64)v5, (__int64)&v77, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v16 + 88) + 16LL))(
              *(_QWORD *)(v16 + 88),
              v17,
              v75,
              *(_QWORD *)(v16 + 56),
              *(_QWORD *)(v16 + 64));
            v33 = *(_QWORD *)v16;
            v34 = *(_QWORD *)v16 + 16LL * *(unsigned int *)(v16 + 8);
            while ( v34 != v33 )
            {
              v35 = *(_QWORD *)(v33 + 8);
              v36 = *(_DWORD *)v33;
              v33 += 16;
              sub_B99FD0(v17, v36, v35);
            }
          }
        }
        v79 = 257;
        return (_QWORD *)sub_B504D0((unsigned int)*v4 - 29, v64, v17, (__int64)&v77, 0, 0);
      }
      v29 = *(_BYTE *)v13;
      if ( *(_BYTE *)v13 <= 0x15u )
      {
        v30 = sub_AD4C30(*((_QWORD *)v4 - 4), v5, 0);
        v31 = a1[2].m128i_i64[0];
        v69 = v30;
        v76 = 257;
        if ( v5 == *(__int64 ***)(v12 + 8) )
        {
          v32 = v12;
        }
        else
        {
          v61 = v31;
          v32 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 **))(**(_QWORD **)(v31 + 80) + 120LL))(
                  *(_QWORD *)(v31 + 80),
                  38,
                  v12,
                  v5);
          if ( !v32 )
          {
            v79 = 257;
            v32 = sub_B51D30(38, v12, (__int64)v5, (__int64)&v77, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v61 + 88) + 16LL))(
              *(_QWORD *)(v61 + 88),
              v32,
              v75,
              *(_QWORD *)(v61 + 56),
              *(_QWORD *)(v61 + 64));
            v44 = *(_QWORD *)v61;
            v45 = *(_QWORD *)v61 + 16LL * *(unsigned int *)(v61 + 8);
            if ( *(_QWORD *)v61 != v45 )
            {
              do
              {
                v46 = *(_QWORD *)(v44 + 8);
                v47 = *(_DWORD *)v44;
                v44 += 16;
                sub_B99FD0(v32, v47, v46);
              }
              while ( v45 != v44 );
            }
          }
        }
        v79 = 257;
        return (_QWORD *)sub_B504D0((unsigned int)*v4 - 29, v32, v69, (__int64)&v77, 0, 0);
      }
      if ( v14 == 68 || v14 == 69 )
      {
        v54 = *(_QWORD *)(v12 - 32);
        if ( v54 )
        {
          if ( v5 == *(__int64 ***)(v54 + 8) )
          {
            v55 = (unsigned int **)a1[2].m128i_i64[0];
            v74 = *(_QWORD *)(v12 - 32);
            v79 = 257;
            v56 = sub_A82DA0(v55, v13, (__int64)v5, (__int64)&v77, 0, 0);
            v79 = 257;
            return (_QWORD *)sub_B504D0((unsigned int)*v4 - 29, v74, v56, (__int64)&v77, 0, 0);
          }
        }
      }
      if ( v29 == 68 || v29 == 69 )
      {
        v40 = *(_QWORD *)(v13 - 32);
        if ( v40 )
        {
          if ( v5 == *(__int64 ***)(v40 + 8) )
          {
            v41 = (unsigned int **)a1[2].m128i_i64[0];
            v42 = *((_QWORD *)v4 - 8);
            v72 = v40;
            v79 = 257;
            v43 = sub_A82DA0(v41, v42, (__int64)v5, (__int64)&v77, 0, 0);
            v79 = 257;
            return (_QWORD *)sub_B504D0((unsigned int)*v4 - 29, v43, v72, (__int64)&v77, 0, 0);
          }
        }
      }
      return sub_1103AD0(a1, a2);
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
      return sub_1103AD0(a1, a2);
    case 13:
    case 14:
      if ( *(_BYTE *)v12 != 67 )
        return sub_1103AD0(a1, a2);
      v18 = *(_QWORD *)(v12 - 32);
      if ( !v18 || *(_BYTE *)v13 > 0x15u )
        return sub_1103AD0(a1, a2);
      LODWORD(v78) = v63;
      v19 = v63 - v7;
      if ( v63 > 0x40 )
      {
        v71 = v13;
        sub_C43690((__int64)&v77, v19, 0);
        v13 = v71;
      }
      else
      {
        v77 = (const char *)(v63 - v7);
      }
      if ( *(_BYTE *)v13 == 17 )
      {
        v65 = (unsigned __int8 *)v13;
        v20 = sub_B532C0(v13 + 24, &v77, 37);
        v21 = v65;
        v22 = v20;
      }
      else
      {
        v37 = *(_QWORD *)(v13 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17 > 1 )
          goto LABEL_13;
        v70 = (unsigned __int8 *)v13;
        v38 = sub_AD7630(v13, 0, v11);
        v21 = v70;
        if ( !v38 || *v38 != 17 )
        {
          if ( *(_BYTE *)(v37 + 8) == 17 )
          {
            v57 = *(_DWORD *)(v37 + 32);
            if ( v57 )
            {
              v22 = 0;
              v48 = 0;
              while ( 1 )
              {
                v62 = v48;
                v73 = v21;
                v49 = (_BYTE *)sub_AD69F0(v21, v48);
                if ( !v49 )
                  break;
                v21 = v73;
                v50 = v62;
                if ( *v49 != 13 )
                {
                  if ( *v49 != 17 )
                    break;
                  v51 = sub_B532C0((__int64)(v49 + 24), &v77, 37);
                  v21 = v73;
                  v50 = v62;
                  v22 = v51;
                  if ( !v51 )
                    break;
                }
                v48 = v50 + 1;
                if ( v57 == v48 )
                  goto LABEL_24;
              }
            }
          }
          goto LABEL_13;
        }
        v39 = sub_B532C0((__int64)(v38 + 24), &v77, 37);
        v21 = v70;
        v22 = v39;
      }
LABEL_24:
      if ( !v22 )
      {
LABEL_13:
        if ( (unsigned int)v78 > 0x40 && v77 )
          j_j___libc_free_0_0(v77);
        return sub_1103AD0(a1, a2);
      }
      if ( (unsigned int)v78 > 0x40 && v77 )
      {
        v66 = v21;
        j_j___libc_free_0_0(v77);
        v21 = v66;
      }
      v67 = v21;
      v60 = *(_BYTE **)(a2 - 32);
      v23 = sub_B44E60((__int64)v60);
      v24 = (_BYTE *)sub_96F3F0((__int64)v67, *(_QWORD *)(v18 + 8), 1, a1[5].m128i_i64[1]);
      if ( !v24 )
        return sub_1103AD0(a1, a2);
      v25 = (_BYTE *)sub_AD7180(v24, v67);
      v68 = (unsigned int **)a1[2].m128i_i64[0];
      if ( *v60 == 56 )
      {
        v52 = sub_BD5D20((__int64)v60);
        v79 = 261;
        v78 = v53;
        v77 = v52;
        v28 = sub_920F70(v68, (_BYTE *)v18, v25, (__int64)&v77, v23);
      }
      else
      {
        v26 = sub_BD5D20((__int64)v60);
        v79 = 261;
        v78 = v27;
        v77 = v26;
        v28 = sub_F94560((__int64 *)v68, v18, (__int64)v25, (__int64)&v77, v23);
      }
      v79 = 257;
      return (_QWORD *)sub_B52120(v28, (__int64)v5, (__int64)&v77, 0, 0);
    default:
      return 0;
  }
}
