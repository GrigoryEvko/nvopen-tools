// Function: sub_1FDB330
// Address: 0x1fdb330
//
__int64 __fastcall sub_1FDB330(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // rax
  unsigned int v8; // eax
  __int64 (*v9)(); // rax
  unsigned int v10; // r13d
  __int64 v12; // rdi
  unsigned int v13; // r13d
  __int64 v14; // rsi
  __int64 v15; // rsi
  unsigned int v16; // edx
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // r13
  char v20; // al
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rdi
  __int64 *v24; // rsi
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // r14
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rsi
  __int32 v33; // eax
  unsigned __int8 v34; // al
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  _QWORD *v39; // rax
  __int64 v40; // rsi
  __int32 v41; // r9d
  int v42; // edx
  int v43; // edx
  int v44; // ecx
  unsigned int v45; // eax
  __int64 v46; // rsi
  unsigned int v47; // eax
  __m128i v48; // [rsp-58h] [rbp-58h] BYREF
  __int64 v49; // [rsp-48h] [rbp-48h]
  _QWORD *v50; // [rsp-40h] [rbp-40h]
  __int64 v51; // [rsp-38h] [rbp-38h]
  char v52; // [rsp-30h] [rbp-30h]

  v5 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v5 + 16) )
    BUG();
  v8 = *(_DWORD *)(v5 + 36);
  if ( v8 <= 0x52 )
  {
    if ( v8 > 0x23 )
    {
      switch ( v8 )
      {
        case '$':
          if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1[5] + 8) + 32LL) + 1744LL) )
            return 1;
          v30 = sub_1601A30(a2, 1);
          v31 = v30;
          if ( !v30 )
            return 1;
          if ( *(_BYTE *)(v30 + 16) == 9 )
            return 1;
          v32 = sub_164A190(v30);
          if ( *(_BYTE *)(v32 + 16) == 17 && (unsigned int)sub_1FDEA40(a1[5], v32) != 0x7FFFFFFF )
            return 1;
          v52 = 0;
          v33 = sub_1FD4C00((__int64)a1, v31);
          if ( v33 )
            goto LABEL_45;
          if ( !*(_QWORD *)(v31 + 8) )
            return 1;
          v34 = *(_BYTE *)(v31 + 16);
          if ( v34 <= 0x17u )
            return 1;
          v35 = a1[5];
          if ( v34 != 53 )
            goto LABEL_44;
          v42 = *(_DWORD *)(v35 + 360);
          if ( !v42 )
            goto LABEL_44;
          v43 = v42 - 1;
          v44 = 1;
          v45 = v43 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          while ( 2 )
          {
            v46 = *(_QWORD *)(*(_QWORD *)(v35 + 344) + 16LL * v45);
            if ( v31 != v46 )
            {
              if ( v46 != -8 )
              {
                v47 = v44 + v45;
                ++v44;
                v45 = v43 & v47;
                continue;
              }
LABEL_44:
              v33 = sub_1FD4520(v35, (__int64 *)v31);
LABEL_45:
              v48.m128i_i64[0] &= 0xFFFFFFF000000000LL;
              v48.m128i_i32[2] = v33;
              v49 = 0;
              v50 = 0;
              v51 = 0;
              v52 = 1;
              sub_1E1C3C0(
                *(_QWORD *)(a1[5] + 784),
                *(unsigned __int64 **)(a1[5] + 792),
                a1 + 10,
                *(_QWORD *)(a1[13] + 8) + 768LL,
                1,
                &v48,
                *(_QWORD *)(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL),
                *(_QWORD *)(*(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL));
            }
            break;
          }
          return 1;
        case '&':
          v17 = *(_QWORD *)(a1[13] + 8) + 768LL;
          v18 = sub_1601A30(a2, 0);
          v19 = v18;
          if ( v18 )
          {
            v20 = *(_BYTE *)(v18 + 16);
            if ( v20 == 13 )
            {
              v21 = a1[5];
              v22 = a1 + 10;
              v23 = *(_QWORD *)(v21 + 784);
              v24 = *(__int64 **)(v21 + 792);
              if ( *(_DWORD *)(v19 + 32) <= 0x40u )
              {
                v25 = sub_1FD3950(v23, v24, v22, v17);
                v27 = v38;
                v39 = *(_QWORD **)(v19 + 24);
                if ( *(_DWORD *)(v19 + 32) > 0x40u )
                  v39 = (_QWORD *)*v39;
                v48.m128i_i64[0] = 1;
                v49 = 0;
                v50 = v39;
              }
              else
              {
                v48.m128i_i64[0] = 2;
                v49 = 0;
                v25 = sub_1FD3950(v23, v24, v22, v17);
                v27 = v26;
                v50 = (_QWORD *)v19;
              }
            }
            else
            {
              if ( v20 != 14 )
              {
                v40 = v19;
                v10 = 1;
                v41 = sub_1FD4C00((__int64)a1, v40);
                if ( v41 )
                  sub_1E1C1F0(
                    *(_QWORD *)(a1[5] + 784),
                    *(unsigned __int64 **)(a1[5] + 792),
                    a1 + 10,
                    v17,
                    0,
                    v41,
                    *(_QWORD *)(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL),
                    *(_QWORD *)(*(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL));
                return v10;
              }
              v36 = sub_1FD3950(*(_QWORD *)(a1[5] + 784), *(__int64 **)(a1[5] + 792), a1 + 10, v17);
              v48.m128i_i64[0] = 3;
              v49 = 0;
              v25 = v36;
              v27 = v37;
              v50 = (_QWORD *)v19;
            }
            sub_1E1A9C0(v27, v25, &v48);
            v48.m128i_i64[0] = 1;
            v49 = 0;
            v50 = 0;
            sub_1E1A9C0(v27, v25, &v48);
            v28 = *(_QWORD **)(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL);
            v48.m128i_i64[0] = 14;
            v49 = 0;
            v50 = v28;
            sub_1E1A9C0(v27, v25, &v48);
            v29 = *(_QWORD **)(*(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL);
            v48.m128i_i64[0] = 14;
            v49 = 0;
            v50 = v29;
            sub_1E1A9C0(v27, v25, &v48);
            return 1;
          }
          v10 = 1;
          sub_1E1C1F0(
            *(_QWORD *)(a1[5] + 784),
            *(unsigned __int64 **)(a1[5] + 792),
            a1 + 10,
            v17,
            0,
            0,
            *(_QWORD *)(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL));
          break;
        case '(':
          return 1;
        case '8':
          goto LABEL_26;
        case 'P':
        case 'Q':
          return sub_1FD9F10(a1, a2);
        case 'R':
          return sub_1FD99F0(a1, a2, a3, a4, a5);
        default:
          goto LABEL_5;
      }
      return v10;
    }
    if ( v8 != 4 )
      goto LABEL_5;
    return 1;
  }
  if ( v8 == 203 )
    goto LABEL_26;
  if ( v8 > 0xCB )
  {
    if ( v8 == 217 )
      return sub_1FDAF60(a1, a2);
    if ( v8 == 218 )
      return sub_1FDB300(a1, a2);
    goto LABEL_5;
  }
  if ( v8 == 144 )
  {
    v12 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v13 = *(_DWORD *)(v12 + 32);
    if ( v13 <= 0x40 )
      v14 = -(__int64)(*(_QWORD *)(v12 + 24) == 0);
    else
      v14 = -(__int64)(v13 == (unsigned int)sub_16A57B0(v12 + 24));
    v15 = sub_15A0680(*(_QWORD *)a2, v14, 0);
LABEL_24:
    v16 = sub_1FD8F60(a1, v15);
    if ( v16 )
    {
      v10 = 1;
      sub_1FD5CC0((__int64)a1, a2, v16, 1);
      return v10;
    }
    return 0;
  }
  if ( v8 > 0x90 )
  {
    v10 = 1;
    if ( v8 == 191 )
      return v10;
    goto LABEL_5;
  }
  if ( v8 == 115 )
  {
LABEL_26:
    v15 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    goto LABEL_24;
  }
  if ( v8 - 116 <= 1 )
    return 1;
LABEL_5:
  v9 = *(__int64 (**)())(*a1 + 48);
  if ( v9 == sub_1FD34A0 )
    return 0;
  return ((__int64 (__fastcall *)(__int64 *, __int64))v9)(a1, a2);
}
