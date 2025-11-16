// Function: sub_2BE1DD0
// Address: 0x2be1dd0
//
void __fastcall sub_2BE1DD0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // r14d
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rbx
  char v10; // al
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rdx
  __m128i *v14; // r13
  __int64 v15; // rax
  __int8 v16; // r15
  __int64 v17; // rax
  __int8 *v18; // rax
  __int64 *v19; // r13
  __int64 v20; // r15
  char v21; // r13
  __int64 v22; // rcx
  unsigned __int8 v23; // [rsp-59h] [rbp-59h]
  unsigned __int8 v24; // [rsp-59h] [rbp-59h]
  unsigned __int8 v25; // [rsp-59h] [rbp-59h]
  __m128i v26; // [rsp-58h] [rbp-58h] BYREF
  __int64 v27; // [rsp-48h] [rbp-48h]

  v3 = (_BYTE *)(a3 + *(_QWORD *)(a1 + 120));
  if ( !*v3 )
  {
    v4 = (unsigned int)a2;
    v5 = a2;
    v6 = a3;
    v7 = a1;
    while ( 2 )
    {
      *v3 = 1;
      v8 = *(_QWORD *)(v7 + 56);
      v9 = *(_QWORD *)(v8 + 56) + 48 * v6;
      switch ( *(_DWORD *)v9 )
      {
        case 1:
          v12 = (*(_BYTE *)(v8 + 24) & 0x10) == 0;
          v13 = *(_QWORD *)(v9 + 16);
          if ( v12 )
          {
            sub_2BE1DD0(v7, v5, v13, v4);
            v21 = *(_BYTE *)(v7 + 140);
            *(_BYTE *)(v7 + 140) = 0;
            sub_2BE1DD0(v7, v5, *(_QWORD *)(v9 + 8), v22);
            *(_BYTE *)(v7 + 140) |= v21;
          }
          else
          {
            a2 = (unsigned int)v4;
            a1 = v7;
            v24 = v4;
            sub_2BE1DD0(v7, (unsigned int)v4, v13, v4);
            v4 = v24;
            if ( !*(_BYTE *)(v7 + 140) )
              goto LABEL_15;
          }
          return;
        case 2:
          if ( !*(_BYTE *)(v9 + 24) )
          {
            a2 = (unsigned int)v4;
            a1 = v7;
            v25 = v4;
            sub_2BE2160(v7, (unsigned int)v4, v6);
            v6 = *(_QWORD *)(v9 + 8);
            v4 = v25;
            goto LABEL_16;
          }
          if ( !*(_BYTE *)(v7 + 140) )
          {
            sub_2BE1DD0(v7, v5, *(_QWORD *)(v9 + 8), v4);
            if ( !*(_BYTE *)(v7 + 140) )
              sub_2BE2160(v7, v5, v6);
          }
          return;
        case 3:
          sub_2BE21F0(v7, v5, v6, v4);
          return;
        case 4:
          if ( *(_QWORD *)(v7 + 24) == *(_QWORD *)(v7 + 32) && (*(_BYTE *)(v7 + 136) & 0x81) == 0 )
            goto LABEL_15;
          return;
        case 5:
          if ( *(_QWORD *)(v7 + 24) == *(_QWORD *)(v7 + 40) && (*(_BYTE *)(v7 + 136) & 2) == 0 )
            goto LABEL_15;
          return;
        case 6:
          a1 = v7;
          v23 = v4;
          v10 = sub_2BE1CB0(v7);
          goto LABEL_5;
        case 7:
          a2 = *(_QWORD *)(v9 + 16);
          a1 = v7;
          v23 = v4;
          v10 = sub_2BE2600(v7, a2);
LABEL_5:
          v4 = v23;
          if ( v10 != (*(_BYTE *)(v9 + 24) ^ 1) )
            return;
LABEL_15:
          v6 = *(_QWORD *)(v9 + 8);
LABEL_16:
          v3 = (_BYTE *)(v6 + *(_QWORD *)(v7 + 120));
          if ( *v3 )
            return;
          continue;
        case 8:
          v19 = (__int64 *)(*(_QWORD *)v7 + 24LL * *(_QWORD *)(v9 + 16));
          v20 = *v19;
          *v19 = *(_QWORD *)(v7 + 24);
          sub_2BE1DD0(v7, v5, *(_QWORD *)(v9 + 8), v4);
          *v19 = v20;
          return;
        case 9:
          v14 = (__m128i *)(*(_QWORD *)v7 + 24LL * *(_QWORD *)(v9 + 16));
          v26 = _mm_loadu_si128(v14);
          v27 = v14[1].m128i_i64[0];
          v15 = *(_QWORD *)(v7 + 24);
          v16 = v14[1].m128i_i8[0];
          v14[1].m128i_i8[0] = 1;
          v14->m128i_i64[1] = v15;
          sub_2BE1DD0(v7, v5, *(_QWORD *)(v9 + 8), v4);
          v14->m128i_i64[0] = v26.m128i_i64[0];
          v17 = v26.m128i_i64[1];
          v14[1].m128i_i8[0] = v16;
          v14->m128i_i64[1] = v17;
          return;
        case 0xB:
          v18 = *(__int8 **)(v7 + 24);
          if ( v18 != *(__int8 **)(v7 + 40) )
          {
            v26.m128i_i8[0] = *v18;
            if ( !*(_QWORD *)(v9 + 32) )
              sub_4263D6(a1, a2, v8);
            if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *, __int64, __int64))(v9 + 40))(
                   v9 + 16,
                   &v26,
                   v8,
                   v4) )
            {
              v26.m128i_i64[0] = *(_QWORD *)(v9 + 8);
              sub_2BE0D90(v7 + 96, &v26, (const __m128i **)v7);
            }
          }
          return;
        case 0xC:
          v11 = *(_QWORD *)(v7 + 24);
          if ( (v11 != *(_QWORD *)(v7 + 32) || (*(_BYTE *)(v7 + 136) & 0x20) == 0)
            && ((_BYTE)v5 == 1 || v11 == *(_QWORD *)(v7 + 40))
            && !*(_BYTE *)(v7 + 140) )
          {
            *(_BYTE *)(v7 + 140) = 1;
            sub_2BDBCA0(*(unsigned __int64 **)(v7 + 64), (const __m128i **)v7, v8);
          }
          return;
        default:
          return;
      }
    }
  }
}
