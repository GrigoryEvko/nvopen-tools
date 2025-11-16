// Function: sub_3749240
// Address: 0x3749240
//
__int64 __fastcall sub_3749240(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned int v8; // eax
  __int64 v10; // rsi
  unsigned int v11; // eax
  unsigned int v12; // r8d
  __int64 (*v14)(); // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdi
  _QWORD *v18; // rax
  _QWORD *i; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 *v23; // rax
  unsigned __int8 **v24; // r8
  __int64 v25; // rdx
  __int64 (__fastcall *v26)(__int64, __int64, _QWORD *, __int64, unsigned __int8 **); // r9
  __int64 v27; // rcx
  _QWORD *v28; // rdx
  __m128i v29; // [rsp+0h] [rbp-40h] BYREF
  __int64 v30; // [rsp+10h] [rbp-30h]
  __int64 v31; // [rsp+18h] [rbp-28h]

  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 )
    goto LABEL_48;
  if ( *(_BYTE *)v5 )
    goto LABEL_48;
  v6 = *(_QWORD *)(a2 + 80);
  if ( *(_QWORD *)(v5 + 24) != v6 )
    goto LABEL_48;
  v8 = *(_DWORD *)(v5 + 36);
  if ( v8 > 0xD3 )
  {
    if ( v8 == 495 )
      return sub_3748960(a1, a2);
    if ( v8 > 0x1EF )
    {
      if ( v8 == 496 )
        return sub_3748D50(a1, a2);
      goto LABEL_21;
    }
    if ( v8 == 346 )
    {
LABEL_28:
      v10 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
LABEL_12:
      v11 = sub_3746830(a1, v10);
      v12 = 0;
      if ( v11 )
      {
        sub_3742B00((__int64)a1, (_BYTE *)a2, v11, 1);
        return 1;
      }
      return v12;
    }
    if ( v8 > 0x15A )
      goto LABEL_21;
    if ( v8 != 282 )
    {
      if ( v8 == 324 )
        return 1;
      goto LABEL_21;
    }
LABEL_48:
    BUG();
  }
  if ( v8 > 0x9A )
  {
    switch ( v8 )
    {
      case 0x9Bu:
      case 0xABu:
      case 0xD2u:
      case 0xD3u:
        return 1;
      case 0x9Cu:
      case 0x9Du:
        return sub_3747A40(a1, a2);
      case 0x9Eu:
        return sub_3747400(a1, a2, a3, v6, a5);
      case 0xCEu:
        goto LABEL_48;
      case 0xD0u:
        goto LABEL_28;
      default:
        goto LABEL_21;
    }
  }
  if ( v8 > 0x5C )
    goto LABEL_21;
  if ( v8 <= 0x43 )
  {
    if ( v8 > 6 )
    {
      v12 = 1;
      if ( v8 == 11 )
        return v12;
    }
    else if ( v8 > 4 )
    {
      v10 = sub_AD6400(*(_QWORD *)(a2 + 8));
      goto LABEL_12;
    }
LABEL_21:
    v12 = 0;
    v14 = *(__int64 (**)())(*a1 + 48);
    if ( v14 != sub_3740EC0 )
      return ((__int64 (__fastcall *)(__int64 *, __int64, __int64 (*)(), __int64, _QWORD))v14)(
               a1,
               a2,
               sub_3740EC0,
               v6,
               0);
    return v12;
  }
  switch ( v8 )
  {
    case 'D':
    case 'G':
      v15 = sub_B58EB0(a2, 0);
      v16 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      if ( **(_BYTE **)(*(_QWORD *)(a2 - 32 * v16) + 24LL) == 4 )
        v15 = 0;
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD, _QWORD, __int64 *))(*a1 + 128))(
        a1,
        v15,
        *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2 - v16)) + 24LL),
        *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v16)) + 24LL),
        a1 + 10);
      return 1;
    case 'E':
      v17 = a1[5];
      if ( *(_BYTE *)(v17 + 924) )
      {
        v18 = *(_QWORD **)(v17 + 904);
        for ( i = &v18[*(unsigned int *)(v17 + 916)]; i != v18; ++v18 )
        {
          if ( a2 == *v18 )
            return 1;
        }
      }
      else if ( sub_C8CA60(v17 + 896, a2) )
      {
        return 1;
      }
      v23 = (unsigned __int8 *)sub_B58EB0(a2, 0);
      v24 = (unsigned __int8 **)(a1 + 10);
      v25 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v26 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64, unsigned __int8 **))(*a1 + 136);
      v27 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v25)) + 24LL);
      v28 = *(_QWORD **)(*(_QWORD *)(a2 + 32 * (2 - v25)) + 24LL);
      if ( v26 != sub_37425D0 )
      {
        v26((__int64)a1, (__int64)v23, v28, v27, v24);
        return 1;
      }
      if ( v23 && (unsigned int)*v23 - 12 > 1 )
        sub_3742340((__int64)a1, (__int64)v23, v28, v27, v24);
      break;
    case 'F':
      v20 = sub_3740F30(
              *(_QWORD *)(a1[5] + 744),
              *(__int64 **)(a1[5] + 752),
              (__int64)(a1 + 10),
              *(_QWORD *)(a1[15] + 8) - 720LL);
      v29.m128i_i64[0] = 14;
      v21 = (__int64)v20;
      LODWORD(v20) = *(_DWORD *)(a2 + 4);
      v30 = 0;
      v31 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * ((unsigned int)v20 & 0x7FFFFFF)) + 24LL);
      sub_2E8EAD0(v22, v21, &v29);
      return 1;
    case 'I':
      return 1;
    case '[':
    case '\\':
      goto LABEL_28;
    default:
      goto LABEL_21;
  }
  return 1;
}
