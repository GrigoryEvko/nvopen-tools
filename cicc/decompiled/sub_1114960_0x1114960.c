// Function: sub_1114960
// Address: 0x1114960
//
__int64 __fastcall sub_1114960(const __m128i *a1, __int64 a2)
{
  unsigned __int8 *v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rdx
  int v5; // ebx
  int v6; // eax
  __int64 v7; // rdx
  __int64 *v8; // rdx
  __int64 v9; // r12
  _BYTE *v10; // rcx
  __int64 v11; // r14
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rsi
  char v16; // r15
  __int64 v17; // r13
  unsigned int v19; // r15d
  bool v20; // al
  __int64 *v21; // rdx
  __m128i v22; // xmm1
  unsigned __int64 v23; // xmm2_8
  __m128i v24; // xmm3
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r9
  _BYTE *v29; // rax
  unsigned __int8 *v30; // rcx
  __int64 v31; // r15
  _BYTE *v32; // rax
  bool v33; // dl
  unsigned int v34; // ebx
  unsigned int v35; // r13d
  _BYTE *v36; // rax
  char v37; // al
  unsigned int v38; // r15d
  _BYTE *v39; // rax
  int v40; // [rsp-94h] [rbp-94h]
  int v41; // [rsp-94h] [rbp-94h]
  _BYTE *v42; // [rsp-90h] [rbp-90h]
  unsigned __int8 *v43; // [rsp-90h] [rbp-90h]
  unsigned __int8 *v44; // [rsp-90h] [rbp-90h]
  bool v45; // [rsp-90h] [rbp-90h]
  _OWORD v46[2]; // [rsp-88h] [rbp-88h] BYREF
  unsigned __int64 v47; // [rsp-68h] [rbp-68h]
  unsigned __int8 *v48; // [rsp-60h] [rbp-60h]
  __m128i v49; // [rsp-58h] [rbp-58h]
  __int64 v50; // [rsp-48h] [rbp-48h]

  if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
    return 0;
  v2 = *(unsigned __int8 **)(a2 - 64);
  if ( *v2 <= 0x1Cu )
    return 0;
  v3 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v3 > 0x15u )
    return 0;
  if ( sub_AC30F0(*(_QWORD *)(a2 - 32)) )
    goto LABEL_5;
  if ( *(_BYTE *)v3 == 17 )
  {
    v19 = *(_DWORD *)(v3 + 32);
    if ( v19 <= 0x40 )
      v20 = *(_QWORD *)(v3 + 24) == 0;
    else
      v20 = v19 == (unsigned int)sub_C444A0(v3 + 24);
  }
  else
  {
    v31 = *(_QWORD *)(v3 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 )
      return 0;
    v32 = sub_AD7630(v3, 0, v4);
    v33 = 0;
    if ( !v32 || *v32 != 17 )
    {
      if ( *(_BYTE *)(v31 + 8) == 17 )
      {
        v41 = *(_DWORD *)(v31 + 32);
        if ( v41 )
        {
          v38 = 0;
          while ( 1 )
          {
            v45 = v33;
            v39 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v3, v38);
            if ( !v39 )
              break;
            v33 = v45;
            if ( *v39 != 13 )
            {
              if ( *v39 != 17 )
                break;
              v33 = sub_9867B0((__int64)(v39 + 24));
              if ( !v33 )
                break;
            }
            if ( v41 == ++v38 )
            {
              if ( v33 )
                goto LABEL_5;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v34 = *((_DWORD *)v32 + 8);
    if ( v34 <= 0x40 )
    {
      if ( *((_QWORD *)v32 + 3) )
        return 0;
      goto LABEL_5;
    }
    v20 = v34 == (unsigned int)sub_C444A0((__int64)(v32 + 24));
  }
  if ( !v20 )
    return 0;
LABEL_5:
  v5 = sub_B53900(a2);
  v6 = *v2;
  if ( (_BYTE)v6 == 67 )
  {
    v7 = *((_QWORD *)v2 - 4);
    if ( (unsigned __int8)(*(_BYTE *)v7 - 55) <= 1u )
    {
      if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      {
        v8 = *(__int64 **)(v7 - 8);
        v9 = *v8;
        if ( !*v8 )
          return 0;
      }
      else
      {
        v8 = (__int64 *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
        v9 = *v8;
        if ( !*v8 )
          return 0;
      }
      v10 = (_BYTE *)v8[4];
      if ( *v10 <= 0x15u )
        goto LABEL_10;
    }
    return 0;
  }
  if ( (unsigned int)(v6 - 55) > 1 )
  {
    if ( (unsigned int)(v6 - 42) > 0x11 )
      return 0;
  }
  else
  {
    if ( (v2[7] & 0x40) != 0 )
    {
      v21 = (__int64 *)*((_QWORD *)v2 - 1);
      v9 = *v21;
      if ( !*v21 )
        goto LABEL_32;
    }
    else
    {
      v21 = (__int64 *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
      v9 = *v21;
      if ( !*v21 )
        goto LABEL_32;
    }
    v10 = (_BYTE *)v21[4];
    if ( *v10 <= 0x15u )
    {
LABEL_10:
      v11 = *(_QWORD *)(v9 + 8);
      v42 = v10;
      v12 = sub_BCB060(v11);
      v14 = (__int64)v42;
      DWORD2(v46[0]) = v12;
      if ( v12 > 0x40 )
      {
        sub_C43690((__int64)v46, v12 - 1, 0);
        v14 = (__int64)v42;
      }
      else
      {
        *(_QWORD *)&v46[0] = v12 - 1;
      }
      if ( *(_BYTE *)v14 == 17 )
      {
        v15 = (__int64)v46;
        v16 = sub_B532C0(v14 + 24, v46, 32);
        goto LABEL_14;
      }
      v17 = *(_QWORD *)(v14 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
      {
        v43 = (unsigned __int8 *)v14;
        v29 = sub_AD7630(v14, 0, v13);
        v30 = v43;
        if ( v29 && *v29 == 17 )
        {
          v15 = (__int64)v46;
          v16 = sub_B532C0((__int64)(v29 + 24), v46, 32);
LABEL_14:
          if ( v16 )
          {
            if ( DWORD2(v46[0]) > 0x40 )
            {
              if ( *(_QWORD *)&v46[0] )
                j_j___libc_free_0_0(*(_QWORD *)&v46[0]);
            }
            goto LABEL_34;
          }
          goto LABEL_18;
        }
        if ( *(_BYTE *)(v17 + 8) == 17 )
        {
          v40 = *(_DWORD *)(v17 + 32);
          if ( v40 )
          {
            v16 = 0;
            v35 = 0;
            while ( 1 )
            {
              v15 = v35;
              v44 = v30;
              v36 = (_BYTE *)sub_AD69F0(v30, v35);
              if ( !v36 )
                break;
              v30 = v44;
              if ( *v36 != 13 )
              {
                if ( *v36 != 17 )
                  break;
                v15 = (__int64)v46;
                v37 = sub_B532C0((__int64)(v36 + 24), v46, 32);
                v30 = v44;
                v16 = v37;
                if ( !v37 )
                  break;
              }
              if ( v40 == ++v35 )
                goto LABEL_14;
            }
          }
        }
      }
LABEL_18:
      if ( DWORD2(v46[0]) > 0x40 && *(_QWORD *)&v46[0] )
        j_j___libc_free_0_0(*(_QWORD *)&v46[0]);
      return 0;
    }
  }
LABEL_32:
  v15 = (__int64)v2;
  v22 = _mm_loadu_si128(a1 + 7);
  v23 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v24 = _mm_loadu_si128(a1 + 9);
  v25 = a1[10].m128i_i64[0];
  v46[0] = _mm_loadu_si128(a1 + 6);
  v47 = v23;
  v50 = v25;
  v48 = v2;
  v46[1] = v22;
  v49 = v24;
  v26 = sub_1195590(a1, v2, v46, 1);
  v9 = v26;
  if ( !v26 )
    return 0;
  v11 = *(_QWORD *)(v26 + 8);
LABEL_34:
  LOWORD(v47) = 257;
  v27 = sub_AD6530(v11, v15);
  return sub_B52500(53, (v5 != 32) + 39, v9, v27, (__int64)v46, v28, 0, 0);
}
