// Function: sub_10C9CD0
// Address: 0x10c9cd0
//
__int64 __fastcall sub_10C9CD0(__int64 a1, __int64 a2, __int64 a3, unsigned __int32 a4)
{
  const void **v5; // r14
  _BYTE *v6; // r12
  unsigned __int8 v8; // al
  __int64 v9; // r10
  unsigned __int64 v10; // r13
  _QWORD *v11; // rax
  int v12; // eax
  unsigned __int64 v13; // rax
  __int32 v14; // edx
  bool v15; // al
  __int64 v17; // rdx
  _BYTE *v18; // rax
  __int64 v19; // rdx
  _BYTE *v20; // rax
  char v21; // r14
  char v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rdi
  bool v26; // r14
  __int64 v27; // r14
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int32 v30; // eax
  _BYTE *v31; // r14
  __int64 v32; // rbx
  bool v33; // zf
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // rdx
  const __m128i *v37; // rax
  __int64 v38; // rax
  __int32 v39; // eax
  __int64 v40; // r15
  unsigned __int64 *v41; // r15
  int v42; // eax
  __int64 v43; // r14
  __int64 v44; // rdx
  _BYTE *v45; // rax
  unsigned int v46; // ecx
  _BYTE *v47; // rax
  unsigned int v48; // ecx
  char v49; // al
  _BYTE *v50; // rax
  __int64 v51; // rsi
  _BYTE *v52; // r14
  __int32 v53; // ebx
  unsigned int v54; // ecx
  _BYTE *v55; // rax
  unsigned int v56; // ecx
  char v57; // al
  char v58; // al
  _BYTE *v59; // r14
  _BYTE *v60; // rax
  _BYTE *v61; // rax
  int v62; // [rsp+8h] [rbp-C8h]
  __int64 *v63; // [rsp+8h] [rbp-C8h]
  __int64 v64; // [rsp+8h] [rbp-C8h]
  int v65; // [rsp+8h] [rbp-C8h]
  int v66; // [rsp+8h] [rbp-C8h]
  __int64 v67; // [rsp+8h] [rbp-C8h]
  unsigned int v68; // [rsp+10h] [rbp-C0h]
  unsigned __int64 **v69; // [rsp+10h] [rbp-C0h]
  unsigned int v70; // [rsp+10h] [rbp-C0h]
  bool v71; // [rsp+10h] [rbp-C0h]
  unsigned int v72; // [rsp+10h] [rbp-C0h]
  unsigned int v73; // [rsp+10h] [rbp-C0h]
  __int64 v75; // [rsp+20h] [rbp-B0h] BYREF
  __int32 v76; // [rsp+28h] [rbp-A8h]
  __int64 *v77; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v78; // [rsp+38h] [rbp-98h] BYREF
  __int64 v79; // [rsp+40h] [rbp-90h]
  unsigned int v80; // [rsp+48h] [rbp-88h]
  __m128i v81; // [rsp+50h] [rbp-80h] BYREF
  __m128i v82; // [rsp+60h] [rbp-70h] BYREF
  __m128i v83; // [rsp+70h] [rbp-60h]
  __m128i v84; // [rsp+80h] [rbp-50h]
  __int64 v85; // [rsp+90h] [rbp-40h]

  v5 = (const void **)(a2 + 24);
  v6 = (_BYTE *)a2;
  v8 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 17 )
    goto LABEL_2;
  v17 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
  if ( (unsigned int)v17 > 1 || v8 > 0x15u )
    goto LABEL_28;
  v18 = sub_AD7630(a2, 1, v17);
  if ( v18 && *v18 == 17 )
  {
    v5 = (const void **)(v18 + 24);
LABEL_2:
    v9 = a3 + 24;
    if ( *(_BYTE *)a3 == 17 )
      goto LABEL_3;
    v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17;
    if ( (unsigned int)v19 <= 1 && *(_BYTE *)a3 <= 0x15u )
    {
      v20 = sub_AD7630(a3, 1, v19);
      if ( v20 )
      {
        if ( *v20 == 17 )
        {
          v9 = (__int64)(v20 + 24);
LABEL_3:
          v10 = a4;
          v68 = *((_DWORD *)v5 + 2);
          if ( v68 > 0x40 )
          {
            v64 = v9;
            v42 = sub_C444A0((__int64)v5);
            v9 = v64;
            if ( v68 - v42 > 0x40 )
              goto LABEL_92;
            v11 = *(_QWORD **)*v5;
          }
          else
          {
            v11 = *v5;
          }
          if ( a4 > (unsigned __int64)v11 )
          {
            if ( *(_DWORD *)(v9 + 8) <= 0x40u )
            {
              v13 = *(_QWORD *)v9;
              goto LABEL_9;
            }
            v62 = *(_DWORD *)(v9 + 8);
            v69 = (unsigned __int64 **)v9;
            v12 = sub_C444A0(v9);
            v9 = (__int64)v69;
            if ( (unsigned int)(v62 - v12) <= 0x40 )
            {
              v13 = **v69;
LABEL_9:
              if ( a4 > v13 )
              {
                LODWORD(v78) = *((_DWORD *)v5 + 2);
                if ( (unsigned int)v78 > 0x40 )
                {
                  v67 = v9;
                  sub_C43780((__int64)&v77, v5);
                  v9 = v67;
                }
                else
                {
                  v77 = (__int64 *)*v5;
                }
                sub_C45EE0((__int64)&v77, (__int64 *)v9);
                v14 = v78;
                LODWORD(v78) = 0;
                v81.m128i_i32[2] = v14;
                v70 = v14;
                v81.m128i_i64[0] = (__int64)v77;
                v63 = v77;
                v15 = sub_D94970((__int64)&v81, (_QWORD *)a4);
                if ( v70 > 0x40 )
                {
                  if ( v63 )
                  {
                    v71 = v15;
                    j_j___libc_free_0_0(v63);
                    v15 = v71;
                    if ( (unsigned int)v78 > 0x40 )
                    {
                      if ( v77 )
                      {
                        j_j___libc_free_0_0(v77);
                        v15 = v71;
                      }
                    }
                  }
                }
                if ( v15 )
                  return sub_AD8D80(*(_QWORD *)(a2 + 8), (__int64)v5);
              }
            }
          }
LABEL_92:
          v8 = *(_BYTE *)a2;
          goto LABEL_29;
        }
      }
    }
  }
  v8 = *(_BYTE *)a2;
LABEL_28:
  v10 = a4;
LABEL_29:
  if ( v8 > 0x15u || *(_BYTE *)a3 > 0x15u )
    goto LABEL_54;
  LODWORD(v78) = a4;
  if ( a4 > 0x40 )
  {
    sub_C43690((__int64)&v77, v10, 0);
    v8 = *(_BYTE *)a2;
  }
  else
  {
    v77 = (__int64 *)v10;
  }
  if ( v8 == 17 )
  {
    v21 = sub_B532C0(a2 + 24, &v77, 36);
    goto LABEL_35;
  }
  v27 = *(_QWORD *)(a2 + 8);
  v28 = (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17;
  if ( (unsigned int)v28 > 1 || v8 > 0x15u )
  {
LABEL_51:
    v26 = 0;
    if ( (unsigned int)v78 <= 0x40 )
      goto LABEL_54;
    goto LABEL_52;
  }
  v35 = sub_AD7630(a2, 0, v28);
  if ( !v35 || *v35 != 17 )
  {
    if ( *(_BYTE *)(v27 + 8) == 17 )
    {
      v65 = *(_DWORD *)(v27 + 32);
      if ( v65 )
      {
        v21 = 0;
        v46 = 0;
        while ( 1 )
        {
          v72 = v46;
          v47 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a2, v46);
          if ( !v47 )
            break;
          v48 = v72;
          if ( *v47 != 13 )
          {
            if ( *v47 != 17 )
              break;
            v49 = sub_B532C0((__int64)(v47 + 24), &v77, 36);
            v48 = v72;
            v21 = v49;
            if ( !v49 )
              break;
          }
          v46 = v48 + 1;
          if ( v65 == v46 )
            goto LABEL_35;
        }
      }
    }
    goto LABEL_51;
  }
  v21 = sub_B532C0((__int64)(v35 + 24), &v77, 36);
LABEL_35:
  if ( !v21 )
    goto LABEL_51;
  v81.m128i_i32[2] = a4;
  if ( a4 > 0x40 )
    sub_C43690((__int64)&v81, v10, 0);
  else
    v81.m128i_i64[0] = v10;
  if ( *(_BYTE *)a3 == 17 )
  {
    v22 = sub_B532C0(a3 + 24, &v81, 36);
  }
  else
  {
    v43 = *(_QWORD *)(a3 + 8);
    v44 = (unsigned int)*(unsigned __int8 *)(v43 + 8) - 17;
    if ( (unsigned int)v44 > 1 || *(_BYTE *)a3 > 0x15u )
    {
LABEL_43:
      v26 = 0;
      goto LABEL_44;
    }
    v45 = sub_AD7630(a3, 0, v44);
    if ( !v45 || *v45 != 17 )
    {
      if ( *(_BYTE *)(v43 + 8) == 17 )
      {
        v66 = *(_DWORD *)(v43 + 32);
        if ( v66 )
        {
          v22 = 0;
          v54 = 0;
          while ( 1 )
          {
            v73 = v54;
            v55 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a3, v54);
            if ( !v55 )
              break;
            v56 = v73;
            if ( *v55 != 13 )
            {
              if ( *v55 != 17 )
                break;
              v57 = sub_B532C0((__int64)(v55 + 24), &v81, 36);
              v56 = v73;
              v22 = v57;
              if ( !v57 )
                break;
            }
            v54 = v56 + 1;
            if ( v66 == v54 )
              goto LABEL_40;
          }
        }
      }
      goto LABEL_43;
    }
    v22 = sub_B532C0((__int64)(v45 + 24), &v81, 36);
  }
LABEL_40:
  if ( !v22 )
    goto LABEL_43;
  v23 = sub_AD57C0(a2, (unsigned __int8 *)a3, 0, 0);
  v25 = (_BYTE *)v23;
  if ( *(_BYTE *)v23 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v23 + 8) + 8LL) - 17 > 1 )
      goto LABEL_43;
    v50 = sub_AD7630(v23, 1, v24);
    v25 = v50;
    if ( !v50 || *v50 != 17 )
      goto LABEL_43;
  }
  v26 = sub_D94970((__int64)(v25 + 24), (_QWORD *)v10);
  if ( !v26 )
    goto LABEL_43;
LABEL_44:
  if ( v81.m128i_i32[2] > 0x40u && v81.m128i_i64[0] )
    j_j___libc_free_0_0(v81.m128i_i64[0]);
  if ( (unsigned int)v78 > 0x40 )
  {
LABEL_52:
    if ( v77 )
      j_j___libc_free_0_0(v77);
  }
  if ( v26 )
    return sub_AD7180((_BYTE *)a2, (unsigned __int8 *)a3);
LABEL_54:
  v29 = *(_QWORD *)(a3 + 16);
  v81.m128i_i64[0] = v10;
  v81.m128i_i64[1] = a2;
  if ( v29 )
  {
    v31 = *(_BYTE **)(v29 + 8);
    if ( !v31
      && *(_BYTE *)a3 == 44
      && sub_F17ED0(&v81, *(_QWORD *)(a3 - 64))
      && *(_QWORD *)(a3 - 32) == v81.m128i_i64[1] )
    {
      v36 = *(_QWORD *)a1;
      v37 = *(const __m128i **)(a1 + 8);
      v81 = _mm_loadu_si128(v37 + 6);
      v82 = _mm_loadu_si128(v37 + 7);
      v83 = _mm_loadu_si128(v37 + 8);
      v84 = _mm_loadu_si128(v37 + 9);
      v38 = v37[10].m128i_i64[0];
      v83.m128i_i64[1] = v36;
      v85 = v38;
      sub_9AC330((__int64)&v77, a2, 0, &v81);
      v39 = v78;
      v40 = (__int64)v77;
      v81.m128i_i32[2] = v78;
      if ( (unsigned int)v78 <= 0x40 )
        goto LABEL_81;
      sub_C43780((__int64)&v81, (const void **)&v77);
      v39 = v81.m128i_i32[2];
      if ( v81.m128i_i32[2] <= 0x40u )
      {
        v40 = v81.m128i_i64[0];
LABEL_81:
        v41 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v39) & ~v40);
        if ( !v39 )
          v41 = 0;
        goto LABEL_83;
      }
      sub_C43D10((__int64)&v81);
      v53 = v81.m128i_i32[2];
      v41 = (unsigned __int64 *)v81.m128i_i64[0];
      v76 = v81.m128i_i32[2];
      v75 = v81.m128i_i64[0];
      if ( v81.m128i_i32[2] <= 0x40u )
      {
LABEL_83:
        if ( (unsigned __int64)v41 < v10 )
          v31 = (_BYTE *)a2;
        goto LABEL_85;
      }
      if ( v53 - (unsigned int)sub_C444A0((__int64)&v75) > 0x40 || *v41 >= v10 )
      {
        if ( !v41 )
          goto LABEL_85;
        v6 = 0;
      }
      v31 = v6;
      j_j___libc_free_0_0(v41);
LABEL_85:
      if ( v80 > 0x40 && v79 )
        j_j___libc_free_0_0(v79);
      if ( (unsigned int)v78 > 0x40 && v77 )
        j_j___libc_free_0_0(v77);
      return (__int64)v31;
    }
  }
  if ( **(_QWORD **)(a1 + 16) != **(_QWORD **)(a1 + 24) || !a4 )
    return 0;
  v30 = a4 - 1;
  v31 = 0;
  if ( ((a4 - 1) & a4) == 0 )
  {
    v32 = v30;
    v33 = *(_BYTE *)a2 == 57;
    v77 = &v75;
    v78 = v30;
    if ( v33 )
    {
      if ( *(_QWORD *)(a2 - 64) )
      {
        v51 = *(_QWORD *)(a2 - 32);
        v75 = *((_QWORD *)v6 - 8);
        if ( sub_F17ED0(&v78, v51) )
        {
          v82.m128i_i64[0] = v32;
          v81.m128i_i64[0] = 0;
          v81.m128i_i64[1] = v75;
          if ( sub_10C9A60((__int64)&v81, 28, (unsigned __int8 *)a3) )
            return v75;
        }
      }
    }
    v33 = *(_BYTE *)a3 == 57;
    v81.m128i_i64[0] = 0;
    v81.m128i_i64[1] = (__int64)v6;
    v82.m128i_i64[0] = v32;
    if ( v33 )
    {
      v52 = *(_BYTE **)(a3 - 64);
      if ( *v52 == 44
        && (unsigned __int8)sub_10081F0((__int64 **)&v81, *((_QWORD *)v52 - 8))
        && *((_QWORD *)v52 - 4) == v81.m128i_i64[1]
        && sub_F17ED0(&v82, *(_QWORD *)(a3 - 32)) )
      {
        return (__int64)v6;
      }
    }
    v33 = *v6 == 68;
    v77 = &v75;
    v78 = v32;
    if ( v33 )
    {
      v34 = (_BYTE *)*((_QWORD *)v6 - 4);
      if ( *v34 == 57 && *((_QWORD *)v34 - 8) )
      {
        v75 = *((_QWORD *)v34 - 8);
        if ( sub_F17ED0(&v78, *((_QWORD *)v34 - 4)) )
        {
          v33 = *(_BYTE *)a3 == 57;
          v81.m128i_i64[0] = 0;
          v82.m128i_i64[0] = v32;
          v81.m128i_i64[1] = v75;
          v82.m128i_i64[1] = v32;
          if ( v33 )
          {
            v59 = *(_BYTE **)(a3 - 64);
            if ( *v59 == 44 )
            {
              if ( (unsigned __int8)sub_10081F0((__int64 **)&v81, *((_QWORD *)v59 - 8)) )
              {
                v60 = (_BYTE *)*((_QWORD *)v59 - 4);
                if ( *v60 == 68 )
                {
                  v61 = (_BYTE *)*((_QWORD *)v60 - 4);
                  if ( *v61 == 57 && *((_QWORD *)v61 - 8) == v81.m128i_i64[1] && sub_F17ED0(&v82, *((_QWORD *)v61 - 4)) )
                  {
                    v31 = v6;
                    if ( sub_F17ED0(&v82.m128i_i64[1], *(_QWORD *)(a3 - 32)) )
                      return (__int64)v31;
                  }
                }
              }
            }
          }
        }
        v58 = *v6;
        v77 = &v75;
        v78 = v32;
        if ( v58 != 68 )
          return 0;
        v34 = (_BYTE *)*((_QWORD *)v6 - 4);
      }
      else
      {
        v77 = &v75;
        v78 = v32;
      }
      if ( *v34 == 57 )
      {
        if ( *((_QWORD *)v34 - 8) )
        {
          v75 = *((_QWORD *)v34 - 8);
          if ( sub_F17ED0(&v78, *((_QWORD *)v34 - 4)) )
          {
            v33 = *(_BYTE *)a3 == 68;
            v81.m128i_i64[0] = 0;
            v82.m128i_i64[0] = v32;
            v81.m128i_i64[1] = v75;
            if ( v33 && sub_10C9A60((__int64)&v81, 28, *(unsigned __int8 **)(a3 - 32)) )
              return (__int64)v6;
          }
        }
      }
    }
    return 0;
  }
  return (__int64)v31;
}
