// Function: sub_3271FA0
// Address: 0x3271fa0
//
__int64 __fastcall sub_3271FA0(__int64 a1, __int64 a2, __int64 a3, unsigned __int16 a4, __m128i *a5)
{
  __int64 v8; // r15
  __int64 result; // rax
  unsigned __int16 v12; // cx
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned __int16 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // r15
  __int64 v21; // rdx
  char v22; // si
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  char v29; // al
  unsigned int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 (__fastcall *v33)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v34; // rax
  int v35; // edx
  __int64 v36; // rax
  unsigned __int16 v37; // [rsp+4h] [rbp-8Ch]
  int v38; // [rsp+8h] [rbp-88h]
  unsigned __int16 v39; // [rsp+8h] [rbp-88h]
  unsigned __int16 v41; // [rsp+10h] [rbp-80h] BYREF
  __int64 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  __int64 v45; // [rsp+30h] [rbp-60h]
  __int64 v46; // [rsp+38h] [rbp-58h]
  __int64 v47; // [rsp+40h] [rbp-50h]
  __int64 v48; // [rsp+48h] [rbp-48h]
  __m128i v49; // [rsp+50h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a2 + 32) > 0x40u )
  {
    v38 = *(_DWORD *)(a2 + 32);
    LODWORD(_R15) = sub_C445E0(a2 + 24);
    if ( !(_DWORD)_R15 || v38 != (_DWORD)_R15 + (unsigned int)sub_C444A0(a2 + 24) )
      return 0;
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 24);
    if ( !v8 || (v8 & (v8 + 1)) != 0 )
      return 0;
    if ( !~v8 )
      goto LABEL_35;
    __asm { tzcnt   r15, r15 }
  }
  switch ( (_DWORD)_R15 )
  {
    case 1:
      v12 = 2;
      goto LABEL_25;
    case 2:
      v12 = 3;
      goto LABEL_25;
    case 4:
      v12 = 4;
      goto LABEL_25;
    case 8:
      v12 = 5;
      goto LABEL_25;
  }
  v12 = 6;
  switch ( (_DWORD)_R15 )
  {
    case 0x10:
      goto LABEL_25;
    case 0x20:
      v12 = 7;
      goto LABEL_25;
    case 0x40:
LABEL_35:
      v12 = 8;
      goto LABEL_25;
    case 0x80:
      v12 = 9;
LABEL_25:
      a5->m128i_i16[0] = v12;
      v16 = 0;
      a5->m128i_i64[1] = 0;
      v17 = *(_WORD *)(a3 + 96);
      v18 = *(_QWORD *)(a3 + 104);
      v41 = v17;
      v42 = v18;
      if ( v12 != v17 )
        goto LABEL_29;
      goto LABEL_26;
  }
  v13 = sub_3007020(*(_QWORD **)(*(_QWORD *)a1 + 64LL), _R15);
  a5->m128i_i16[0] = v13;
  v15 = v14;
  v16 = v14;
  v12 = v13;
  a5->m128i_i64[1] = v14;
  v17 = *(_WORD *)(a3 + 96);
  v18 = *(_QWORD *)(a3 + 104);
  v41 = v17;
  v42 = v18;
  if ( v17 != v13 )
    goto LABEL_29;
  if ( !v17 )
  {
    if ( v18 == v15 && !*(_BYTE *)(a1 + 33) )
      return 1;
    goto LABEL_29;
  }
LABEL_26:
  result = *(unsigned __int8 *)(a1 + 33);
  if ( !(_BYTE)result )
    return 1;
  if ( a4 && (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 2 * (v12 + 274LL * a4 + 71704) + 7) & 0xF0) == 0 )
    return result;
LABEL_29:
  if ( (*(_BYTE *)(*(_QWORD *)(a3 + 112) + 37LL) & 0xF) != 0 || (*(_BYTE *)(a3 + 32) & 8) != 0 )
    return 0;
  if ( v17 == v12 )
  {
    if ( v17 || v16 == v18 )
      return 0;
    v49.m128i_i64[1] = v16;
    v49.m128i_i16[0] = 0;
    goto LABEL_38;
  }
  v49.m128i_i16[0] = v12;
  v49.m128i_i64[1] = v16;
  if ( !v12 )
  {
LABEL_38:
    v39 = v12;
    v19 = sub_3007260((__int64)&v49);
    v12 = v39;
    v45 = v19;
    v20 = v19;
    v46 = v21;
    v22 = v21;
    goto LABEL_39;
  }
  if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
    goto LABEL_83;
  v20 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
  v22 = byte_444C4A0[16 * v12 - 8];
LABEL_39:
  if ( v17 )
  {
    if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      goto LABEL_83;
    v26 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
    result = (unsigned __int8)byte_444C4A0[16 * v17 - 8];
  }
  else
  {
    v37 = v12;
    v23 = sub_3007260((__int64)&v41);
    v12 = v37;
    v25 = v24;
    v43 = v23;
    v26 = v23;
    v44 = v25;
    result = (unsigned __int8)v25;
  }
  if ( (_BYTE)result || !v22 )
  {
    if ( v26 <= v20 )
      return 0;
    if ( !v12 )
    {
      if ( sub_3007100((__int64)a5) )
        return 0;
      v47 = sub_3007260((__int64)a5);
      v48 = v27;
      v28 = v47;
      v29 = v48;
      goto LABEL_47;
    }
    if ( (unsigned __int16)(v12 - 176) <= 0x34u )
      return 0;
    if ( v12 != 1 && (unsigned __int16)(v12 - 504) > 7u )
    {
      v36 = 16LL * (v12 - 1);
      v28 = *(_QWORD *)&byte_444C4A0[v36];
      v29 = byte_444C4A0[v36 + 8];
LABEL_47:
      v49.m128i_i64[0] = v28;
      v49.m128i_i8[8] = v29;
      v30 = sub_CA1930(&v49);
      if ( v30 > 7 && (v30 & (v30 - 1)) == 0 )
      {
        v31 = *(_QWORD *)(a1 + 8);
        if ( !*(_BYTE *)(a1 + 33)
          || (v32 = a5->m128i_u16[0], a4)
          && (_WORD)v32
          && (*(_BYTE *)(v31 + 2 * (v32 + 274LL * a4 + 71704) + 7) & 0xF0) == 0 )
        {
          v33 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v31 + 776LL);
          if ( v33 != sub_2FE41D0 )
            return v33(v31, a3, 3, a5->m128i_u32[0], a5->m128i_i64[1]);
          v49 = _mm_loadu_si128(a5);
          if ( v49.m128i_i16[0] )
          {
            if ( (unsigned __int16)(v49.m128i_i16[0] - 17) <= 0xD3u )
            {
LABEL_56:
              v34 = *(_QWORD *)(a3 + 56);
              if ( v34 )
              {
                v35 = 1;
                do
                {
                  if ( !*(_DWORD *)(v34 + 8) )
                  {
                    if ( !v35 )
                      return 0;
                    v34 = *(_QWORD *)(v34 + 32);
                    if ( !v34 )
                      return 1;
                    if ( !*(_DWORD *)(v34 + 8) )
                      return 0;
                    v35 = 0;
                  }
                  v34 = *(_QWORD *)(v34 + 32);
                }
                while ( v34 );
                if ( v35 != 1 )
                  return 1;
              }
              return 0;
            }
          }
          else if ( sub_30070B0((__int64)&v49) )
          {
            goto LABEL_56;
          }
          return 1;
        }
      }
      return 0;
    }
LABEL_83:
    BUG();
  }
  return result;
}
