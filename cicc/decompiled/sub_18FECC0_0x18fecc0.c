// Function: sub_18FECC0
// Address: 0x18fecc0
//
__int64 __fastcall sub_18FECC0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // r13d
  __m128i v6; // xmm7
  __m128i v7; // xmm6
  __m128i v8; // xmm1
  unsigned int v9; // edx
  __m128i v10; // xmm3
  __m128i v11; // xmm5
  __int64 *v12; // [rsp+18h] [rbp-98h] BYREF
  __m128i v13; // [rsp+20h] [rbp-90h] BYREF
  __m128i v14; // [rsp+30h] [rbp-80h] BYREF
  __int64 v15; // [rsp+40h] [rbp-70h]
  __m128i v16; // [rsp+50h] [rbp-60h] BYREF
  __m128i v17; // [rsp+60h] [rbp-50h] BYREF
  __int64 v18; // [rsp+70h] [rbp-40h]

  switch ( *(_BYTE *)(a2 + 16) )
  {
    case '6':
      sub_141EB40(&v13, (__int64 *)a2);
      goto LABEL_10;
    case '7':
      sub_141EDF0(&v13, a2);
      goto LABEL_12;
    case ':':
      sub_141F110(&v13, a2);
LABEL_10:
      v10 = _mm_loadu_si128(&v14);
      v16 = _mm_loadu_si128(&v13);
      v18 = v15;
      v17 = v10;
      goto LABEL_5;
    case ';':
      sub_141F3C0(&v13, a2);
LABEL_12:
      v11 = _mm_loadu_si128(&v14);
      v16 = _mm_loadu_si128(&v13);
      v18 = v15;
      v17 = v11;
      goto LABEL_5;
    case 'R':
      sub_141F0A0(&v13, a2);
      v6 = _mm_loadu_si128(&v14);
      v7 = _mm_loadu_si128(&v13);
      v18 = v15;
      v16 = v7;
      v17 = v6;
LABEL_5:
      v8 = _mm_loadu_si128(&v17);
      v13 = _mm_loadu_si128(&v16);
      v15 = v18;
      v14 = v8;
      v4 = sub_18FEB70(a1 + 392, v13.m128i_i64, &v12);
      if ( (_BYTE)v4 )
      {
        v9 = sub_18FEB70(a1 + 392, v13.m128i_i64, &v12);
        if ( (_BYTE)v9 )
        {
          if ( v12 == (__int64 *)(*(_QWORD *)(a1 + 400) + 48LL * *(unsigned int *)(a1 + 416)) )
            v4 = v9;
          else
            LOBYTE(v4) = a3 >= *(_DWORD *)(v12[5] + 56);
        }
      }
      break;
    default:
      v4 = 0;
      break;
  }
  return v4;
}
