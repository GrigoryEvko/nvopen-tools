// Function: sub_8955E0
// Address: 0x8955e0
//
__int64 __fastcall sub_8955E0(__m128i *a1, _DWORD *a2)
{
  _QWORD *v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 i; // rax
  __int64 j; // rbx
  __int64 v12; // r14
  const __m128i *v13; // r15
  __int64 result; // rax
  char v15; // al
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 *v21; // r9
  __int64 v22; // r8
  __int64 v23; // rdi
  int v24; // edx
  __int64 v25; // [rsp-10h] [rbp-D0h]
  __int64 v26; // [rsp-8h] [rbp-C8h]
  char v27; // [rsp+Fh] [rbp-B1h]
  __int64 v28; // [rsp+10h] [rbp-B0h]
  __int64 v29; // [rsp+10h] [rbp-B0h]
  __int64 v30; // [rsp+10h] [rbp-B0h]
  _QWORD *v31; // [rsp+18h] [rbp-A8h]
  _QWORD *v32; // [rsp+18h] [rbp-A8h]
  _QWORD *v33; // [rsp+18h] [rbp-A8h]
  unsigned int v34; // [rsp+2Ch] [rbp-94h] BYREF
  __m128i v35[5]; // [rsp+30h] [rbp-90h] BYREF
  int v36; // [rsp+80h] [rbp-40h]

  v4 = (_QWORD *)a1->m128i_i64[1];
  v5 = *(_QWORD *)(*(_QWORD *)(*v4 + 96LL) + 32LL);
  v6 = *(_QWORD *)(v5 + 88);
  v7 = *(_QWORD *)(v6 + 88);
  if ( v7 )
  {
    if ( (*(_BYTE *)(v6 + 160) & 1) != 0 )
      v7 = *(_QWORD *)(*(_QWORD *)(*v4 + 96LL) + 32LL);
  }
  else
  {
    v7 = *(_QWORD *)(*(_QWORD *)(*v4 + 96LL) + 32LL);
  }
  switch ( *(_BYTE *)(v7 + 80) )
  {
    case 4:
    case 5:
      v8 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 80LL);
      break;
    case 6:
      v8 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v8 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v8 = *(_QWORD *)(v7 + 88);
      break;
    default:
      BUG();
  }
  v9 = *(__int64 **)(v8 + 176);
  for ( i = v9[19]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = v4[19]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v12 = *(_QWORD *)(j + 168);
  v13 = *(const __m128i **)(*(_QWORD *)(i + 168) + 56LL);
  result = v13->m128i_u8[0];
  if ( (result & 0x20) != 0 )
  {
    v30 = *(_QWORD *)(*(_QWORD *)(*v4 + 96LL) + 32LL);
    v33 = (_QWORD *)a1->m128i_i64[1];
    sub_894C00(*v9);
    result = v13->m128i_u8[0];
    v5 = v30;
    v4 = v33;
  }
  if ( (result & 0x40) == 0 || (v29 = v5, v32 = v4, result = sub_8955E0(a1, a2), v4 = v32, v5 = v29, !a2) || !*a2 )
  {
    *a1 = _mm_loadu_si128(v13);
    a1[1] = _mm_loadu_si128(v13 + 1);
    if ( (v13->m128i_i8[0] & 1) != 0 )
    {
      result = a1->m128i_i64[1];
      v28 = v5;
      v31 = v4;
      if ( result )
      {
        v15 = *(_BYTE *)(result + 173);
        v34 = 0;
        v27 = v15;
        sub_892150(v35);
        v36 = 1;
        sub_865900(v28);
        sub_88FF90((__int64)v35, **(__int64 ***)(j + 168));
        v16 = *(_QWORD *)(v31[5] + 32LL);
        sub_744F60((__m128i **)&a1->m128i_i64[1], v16, 0, 0, 0x840u, v35, (_DWORD *)v31 + 16, (int *)&v34);
        v17 = v35[0].m128i_i64[0];
        sub_8921C0(v35[0].m128i_i64[0]);
        sub_864110(v17, v16, v18, v19, v20, v21);
        result = v25;
        if ( v34 )
        {
          if ( a2 )
          {
            *a2 = 1;
            return result;
          }
          if ( v27 )
          {
            sub_6851C0(0xB31u, (_DWORD *)v31 + 16);
            result = sub_72C970(a1->m128i_i64[1]);
          }
        }
        else
        {
          v23 = a1->m128i_i64[1];
          if ( *(_BYTE *)(v23 + 173) != 12 )
          {
            v24 = 4 * (sub_711520(v23, v16, v26, v34, v22) & 1);
            result = v24 | a1->m128i_i8[0] & 0xFBu;
            a1->m128i_i8[0] = v24 | a1->m128i_i8[0] & 0xFB;
          }
        }
      }
    }
    *(_QWORD *)(v12 + 56) = a1;
  }
  return result;
}
