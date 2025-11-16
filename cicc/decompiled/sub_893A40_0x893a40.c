// Function: sub_893A40
// Address: 0x893a40
//
_QWORD *__fastcall sub_893A40(__int64 a1, __m128i *a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r14
  __int64 v9; // rdi
  _QWORD *v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdi
  __m128i *v16; // rdi
  __int8 v17; // dl
  __int8 v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __m128i *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r13
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-60h]
  __int64 *v34; // [rsp+10h] [rbp-50h]
  _DWORD v35[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = *(_QWORD *)a1;
  v6 = *(_QWORD *)(a1 + 336);
  v7 = *(_QWORD *)(*(_QWORD *)a1 + 304LL);
  if ( word_4F06418[0] == 163
    || word_4F06418[0] == 73
    || word_4F06418[0] == 55
    || word_4F06418[0] == 56 && (unsigned int)sub_651030(v35) )
  {
    sub_685360(0xB40u, (_DWORD *)(v5 + 32), *(_QWORD *)(v5 + 288));
  }
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 168) + 32LL);
  sub_878710(v8, a2);
  a2->m128i_i64[1] = *(_QWORD *)(v5 + 32);
  if ( *(_DWORD *)(v8 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * *(int *)(a1 + 204)) )
  {
    if ( (*(_BYTE *)(v8 + 81) & 0x10) != 0
      && (unsigned __int8)sub_87D550(v8) != (*(_BYTE *)(qword_4F04C68[0] + 776LL * *(int *)(a1 + 204) + 5) & 3) )
    {
      sub_6854C0(0xB4Fu, (FILE *)(v5 + 32), v8);
    }
  }
  else
  {
    sub_6854C0(0xB41u, (FILE *)(v5 + 32), v8);
    *(_DWORD *)(a1 + 52) = 1;
    *a2 = _mm_loadu_si128(xmmword_4F06660);
    a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v32 = *(_QWORD *)dword_4F07508;
    a2[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    a2[1].m128i_i8[1] |= 0x20u;
    a2->m128i_i64[1] = v32;
  }
  v9 = *(_QWORD *)(v5 + 288);
  *(_BYTE *)(v5 + 122) = ((*(_BYTE *)(a3 + 64) & 4) != 0) | *(_BYTE *)(v5 + 122) & 0xFE;
  sub_8DCB20(v9);
  *(_QWORD *)(a1 + 440) = *(_QWORD *)a3;
  *(_QWORD *)(a1 + 448) = *(_QWORD *)(a3 + 8);
  v10 = sub_87EBB0(0x14u, a2->m128i_i64[0], &a2->m128i_i64[1]);
  sub_88DD80(a1, (__int64)v10, v11, v12, v13, v14);
  *((_DWORD *)v10 + 10) = *(_DWORD *)(qword_4F04C68[0] + 776LL * *(int *)(a1 + 204));
  switch ( *((_BYTE *)v10 + 80) )
  {
    case 4:
    case 5:
      v15 = *(_QWORD *)(v10[12] + 80LL);
      break;
    case 6:
      v15 = *(_QWORD *)(v10[12] + 32LL);
      break;
    case 9:
    case 0xA:
      v15 = *(_QWORD *)(v10[12] + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v15 = v10[11];
      break;
    default:
      MEMORY[0xA0] &= ~8u;
      BUG();
  }
  v16 = (__m128i *)(v15 + 296);
  v17 = (8 * (*(_BYTE *)(a1 + 84) & 1)) | v16[-9].m128i_i8[8] & 0xF7;
  v16[-9].m128i_i8[8] = v17;
  v18 = v17 & 0xEF | (16 * (*(_BYTE *)(a1 + 88) & 1));
  v16[-9].m128i_i8[8] = v18;
  v16[-9].m128i_i8[8] = (32 * (*(_BYTE *)(a1 + 128) & 1)) | v18 & 0xDF;
  sub_879080(v16, 0, *(_QWORD *)(a1 + 192));
  *(_QWORD *)(v6 + 200) = v6;
  *(_QWORD *)(v6 + 208) = v6;
  v19 = v10[11];
  *(_QWORD *)(v19 + 104) = v6;
  v33 = v19;
  sub_877D80(v6, v10);
  sub_877F10(v6, (__int64)v10, v20, v21, v22, v23);
  v24 = sub_725FD0();
  v25 = *(_QWORD *)(v5 + 288);
  v24[9].m128i_i64[1] = v25;
  v24[16].m128i_i64[1] = v25;
  sub_65BB50(v5, v8);
  sub_725ED0((__int64)v24, 7);
  v24[11].m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v8 + 88) + 104LL);
  *(_QWORD *)(*(_QWORD *)(v24[9].m128i_i64[1] + 168) + 8LL) = v24;
  *(_QWORD *)(v33 + 176) = v24;
  v34 = sub_880CF0((__int64)v10, (__int64)v24, **(_QWORD **)(a1 + 192));
  sub_877D80((__int64)v24, v34);
  sub_877F10((__int64)v24, (__int64)v34, v26, v27, v28, v29);
  v24[5].m128i_i8[8] = v24[5].m128i_i8[8] & 0x8F | 0x20;
  sub_8CCE20(v34, v33);
  sub_65BAF0(v5, (__int64)v24);
  if ( dword_4F07590 && (a2[1].m128i_i8[1] & 0x20) == 0 )
    sub_7362F0((__int64)v24, -1);
  if ( !dword_4F04C3C )
  {
    *(_QWORD *)(v5 + 352) = *(_QWORD *)(*(_QWORD *)(a1 + 336) + 96LL);
    sub_65C210(v5);
  }
  switch ( *(_BYTE *)(v8 + 80) )
  {
    case 4:
    case 5:
      v30 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 80LL);
      goto LABEL_17;
    case 6:
      v30 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 32LL);
      goto LABEL_17;
    case 9:
    case 0xA:
      v30 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 56LL);
      if ( (a2[1].m128i_i8[1] & 0x20) == 0 )
        goto LABEL_20;
      goto LABEL_18;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v30 = *(_QWORD *)(v8 + 88);
      goto LABEL_17;
    default:
      v30 = 0;
LABEL_17:
      if ( (a2[1].m128i_i8[1] & 0x20) == 0 )
      {
LABEL_20:
        sub_87F0B0((__int64)v10, (__int64 *)(v30 + 216));
        *(_BYTE *)(v30 + 266) |= 0x80u;
      }
LABEL_18:
      sub_893800(a1, (__int64)v10, v33);
      return v10;
  }
}
