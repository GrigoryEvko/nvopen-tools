// Function: sub_2BE13A0
// Address: 0x2be13a0
//
void __fastcall sub_2BE13A0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 v6; // r12
  unsigned __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rbx
  char v10; // al
  unsigned __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rdx
  __m128i *v14; // r13
  __int64 v15; // rax
  __int8 v16; // r15
  __int64 v17; // rax
  __int8 *v18; // rax
  __int64 v19; // rcx
  __int64 *v20; // r13
  __int64 v21; // r15
  char v22; // r13
  __int64 v23; // rcx
  __int64 v24; // rcx
  unsigned __int8 v25; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v26; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v27; // [rsp+Fh] [rbp-51h]
  __m128i v28; // [rsp+10h] [rbp-50h] BYREF
  __int64 v29; // [rsp+20h] [rbp-40h]

  v3 = a2;
  v6 = a1;
  v7 = *(_QWORD *)(a1 + 56);
  v8 = *(_QWORD *)(v7 + 56);
  while ( 2 )
  {
    v9 = v8 + 48 * a3;
    switch ( *(_DWORD *)v9 )
    {
      case 1:
        v12 = (*(_BYTE *)(v7 + 24) & 0x10) == 0;
        v13 = *(_QWORD *)(v9 + 16);
        if ( v12 )
        {
          sub_2BE13A0(v6, a2, v13, v3);
          v22 = *(_BYTE *)(v6 + 116);
          *(_BYTE *)(v6 + 116) = 0;
          sub_2BE13A0(v6, a2, *(_QWORD *)(v9 + 8), v23);
          *(_BYTE *)(v6 + 116) |= v22;
        }
        else
        {
          a1 = v6;
          v26 = v3;
          sub_2BE13A0(v6, (unsigned int)v3, v13, v3);
          v3 = v26;
          if ( !*(_BYTE *)(v6 + 116) )
            goto LABEL_15;
        }
        return;
      case 2:
        if ( *(_BYTE *)(v9 + 24) )
        {
          sub_2BE13A0(v6, a2, *(_QWORD *)(v9 + 8), v3);
          if ( !*(_BYTE *)(v6 + 116) )
            sub_2BE1900(v6, a2, a3);
        }
        else
        {
          a1 = v6;
          v27 = v3;
          sub_2BE1900(v6, (unsigned int)v3, a3);
          v3 = v27;
          if ( !*(_BYTE *)(v6 + 116) )
            goto LABEL_15;
        }
        return;
      case 3:
        sub_2BE1730(v6, a2, a3, v3);
        return;
      case 4:
        if ( *(_QWORD *)(v6 + 24) == *(_QWORD *)(v6 + 32) && (*(_BYTE *)(v6 + 112) & 0x81) == 0 )
          goto LABEL_25;
        return;
      case 5:
        if ( *(_QWORD *)(v6 + 24) == *(_QWORD *)(v6 + 40) && (*(_BYTE *)(v6 + 112) & 2) == 0 )
        {
LABEL_25:
          a3 = *(_QWORD *)(v9 + 8);
          continue;
        }
        return;
      case 6:
        a1 = v6;
        v25 = v3;
        v10 = sub_2BE1280(v6);
        goto LABEL_4;
      case 7:
        a1 = v6;
        v25 = v3;
        v10 = sub_2BE1990(v6, *(_QWORD *)(v9 + 16));
LABEL_4:
        v3 = v25;
        if ( v10 != (*(_BYTE *)(v9 + 24) ^ 1) )
          return;
LABEL_15:
        v7 = *(_QWORD *)(v6 + 56);
        a3 = *(_QWORD *)(v9 + 8);
        v8 = *(_QWORD *)(v7 + 56);
        continue;
      case 8:
        v20 = (__int64 *)(*(_QWORD *)v6 + 24LL * *(_QWORD *)(v9 + 16));
        v21 = *v20;
        *v20 = *(_QWORD *)(v6 + 24);
        sub_2BE13A0(v6, a2, *(_QWORD *)(v9 + 8), v3);
        *v20 = v21;
        return;
      case 9:
        v14 = (__m128i *)(*(_QWORD *)v6 + 24LL * *(_QWORD *)(v9 + 16));
        v28 = _mm_loadu_si128(v14);
        v29 = v14[1].m128i_i64[0];
        v15 = *(_QWORD *)(v6 + 24);
        v16 = v14[1].m128i_i8[0];
        v14[1].m128i_i8[0] = 1;
        v14->m128i_i64[1] = v15;
        sub_2BE13A0(v6, a2, *(_QWORD *)(v9 + 8), v3);
        v14->m128i_i64[0] = v28.m128i_i64[0];
        v17 = v28.m128i_i64[1];
        v14[1].m128i_i8[0] = v16;
        v14->m128i_i64[1] = v17;
        return;
      case 0xB:
        v18 = *(__int8 **)(v6 + 24);
        if ( v18 != *(__int8 **)(v6 + 40) )
        {
          v28.m128i_i8[0] = *v18;
          if ( !*(_QWORD *)(v9 + 32) )
            sub_4263D6(a1, v8, v7);
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *, unsigned __int64, __int64))(v9 + 40))(
                 v9 + 16,
                 &v28,
                 v7,
                 v3) )
          {
            ++*(_QWORD *)(v6 + 24);
            sub_2BE13A0(v6, a2, *(_QWORD *)(v9 + 8), v19);
            --*(_QWORD *)(v6 + 24);
          }
        }
        return;
      case 0xC:
        v11 = *(_QWORD *)(v6 + 24);
        if ( (_BYTE)a2 )
        {
          *(_BYTE *)(v6 + 116) = 1;
          if ( v11 == *(_QWORD *)(v6 + 32) && (*(_BYTE *)(v6 + 112) & 0x20) != 0 )
          {
LABEL_40:
            *(_BYTE *)(v6 + 116) = 0;
            return;
          }
        }
        else
        {
          v24 = *(_QWORD *)(v6 + 40);
          *(_BYTE *)(v6 + 116) = v11 == v24;
          if ( v11 == *(_QWORD *)(v6 + 32) && (*(_BYTE *)(v6 + 112) & 0x20) != 0 )
            goto LABEL_40;
          if ( v11 != v24 )
            return;
        }
        if ( (*(_BYTE *)(v7 + 24) & 0x10) != 0 )
        {
LABEL_12:
          sub_2BDBCA0(*(unsigned __int64 **)(v6 + 64), (const __m128i **)v6, v7);
          return;
        }
        v7 = *(_QWORD *)(v6 + 104);
        if ( v11 > v7 || !v7 )
        {
          *(_QWORD *)(v6 + 104) = *(_QWORD *)(v6 + 24);
          goto LABEL_12;
        }
        return;
      default:
        return;
    }
  }
}
