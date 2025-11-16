// Function: sub_117B470
// Address: 0x117b470
//
unsigned __int8 *__fastcall sub_117B470(const __m128i **a1, __int64 a2, unsigned __int8 *a3, __int64 *a4, char a5)
{
  __int64 v8; // rbx
  unsigned int v9; // ecx
  __int64 v10; // rsi
  unsigned __int8 *v11; // rax
  __int64 v12; // rdx
  unsigned __int8 *v13; // r15
  unsigned __int8 v14; // al
  unsigned __int8 *v15; // rax
  unsigned int v16; // ebx
  unsigned __int8 *v17; // r8
  int v18; // eax
  bool v19; // al
  int v20; // eax
  bool v21; // al
  bool v22; // al
  unsigned int v23; // ebx
  bool v24; // al
  unsigned __int8 *v25; // r14
  __int64 v27; // rdx
  _BYTE *v28; // rax
  unsigned int **v29; // rdi
  unsigned __int8 *v30; // rax
  unsigned __int8 *v31; // r15
  int v32; // edi
  int v33; // ebx
  int v34; // esi
  bool v35; // al
  const __m128i *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // eax
  unsigned __int64 v40; // rdx
  __int16 v41; // ax
  int v42; // [rsp+8h] [rbp-C8h]
  __int64 v43; // [rsp+8h] [rbp-C8h]
  unsigned __int8 *v44; // [rsp+10h] [rbp-C0h]
  unsigned __int8 *v45; // [rsp+10h] [rbp-C0h]
  __int64 v46; // [rsp+18h] [rbp-B8h]
  char v47; // [rsp+18h] [rbp-B8h]
  unsigned int v48; // [rsp+24h] [rbp-ACh]
  unsigned __int8 v49; // [rsp+2Bh] [rbp-A5h]
  unsigned __int8 *v51; // [rsp+38h] [rbp-98h]
  unsigned __int64 v52; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v53; // [rsp+48h] [rbp-88h]
  __m128i v54[2]; // [rsp+50h] [rbp-80h] BYREF
  __m128i v55; // [rsp+70h] [rbp-60h]
  __m128i v56; // [rsp+80h] [rbp-50h]
  __int64 v57; // [rsp+90h] [rbp-40h]

  switch ( *a3 )
  {
    case '*':
    case '+':
    case '.':
    case '/':
    case '9':
    case ':':
    case ';':
      if ( *((__int64 **)a3 - 8) == a4 )
      {
        v8 = -32;
        goto LABEL_3;
      }
      v8 = -64;
      if ( a4 == *((__int64 **)a3 - 4) )
        goto LABEL_3;
      return 0;
    case ',':
    case '-':
    case '2':
    case '6':
    case '7':
    case '8':
      v8 = -32;
      if ( a4 != *((__int64 **)a3 - 8) )
        return 0;
LABEL_3:
      v49 = sub_920620(a2);
      if ( v49 )
      {
        v48 = sub_B45210(a2);
        v9 = (v48 >> 3) & 1;
        v49 = (v48 & 8) != 0;
      }
      else
      {
        v48 = 0;
        v9 = 0;
      }
      v10 = *((_QWORD *)a3 + 1);
      v11 = sub_AD93D0((unsigned int)*a3 - 29, v10, 1, v9);
      v13 = *(unsigned __int8 **)&a3[v8];
      v51 = v11;
      v14 = *v13;
      if ( *v13 == 17 )
      {
        v46 = (__int64)(v13 + 24);
      }
      else
      {
        v27 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v13 + 1) + 8LL) - 17;
        if ( (unsigned int)v27 > 1 )
        {
          if ( v14 <= 0x15u )
            return 0;
          goto LABEL_29;
        }
        if ( v14 > 0x15u )
          goto LABEL_29;
        v10 = 0;
        v28 = sub_AD7630((__int64)v13, 0, v27);
        if ( !v28 )
        {
          if ( *v13 <= 0x15u )
            return 0;
          goto LABEL_29;
        }
        v12 = *v13;
        if ( *v28 != 17 )
        {
          if ( (unsigned __int8)v12 <= 0x15u )
            return 0;
          goto LABEL_29;
        }
        v46 = (__int64)(v28 + 24);
        if ( (unsigned __int8)v12 > 0x15u )
          goto LABEL_29;
      }
      v15 = sub_AD8340(v51, v10, v12);
      v16 = *((_DWORD *)v15 + 2);
      v17 = v15;
      if ( v16 <= 0x40 )
      {
        v19 = *(_QWORD *)v15 == 0;
      }
      else
      {
        v44 = v15;
        v18 = sub_C444A0((__int64)v15);
        v17 = v44;
        v19 = v16 == v18;
      }
      if ( !v19 )
      {
        if ( *(_DWORD *)(v46 + 8) <= 0x40u )
        {
          v21 = *(_QWORD *)v46 == 0;
        }
        else
        {
          v42 = *(_DWORD *)(v46 + 8);
          v45 = v17;
          v20 = sub_C444A0(v46);
          v17 = v45;
          v21 = v42 == v20;
        }
        if ( !v21 )
          return 0;
      }
      if ( v16 <= 0x40 )
      {
        if ( *(_QWORD *)v17 == 1 || !v16 )
          goto LABEL_29;
        v22 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) == *(_QWORD *)v17;
      }
      else
      {
        v43 = (__int64)v17;
        if ( v16 - 1 == (unsigned int)sub_C444A0((__int64)v17) )
          goto LABEL_29;
        v22 = v16 == (unsigned int)sub_C445E0(v43);
      }
      if ( v22 )
        goto LABEL_29;
      v23 = *(_DWORD *)(v46 + 8);
      if ( v23 <= 0x40 )
      {
        if ( *(_QWORD *)v46 == 1 || !v23 )
          goto LABEL_29;
        v24 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23) == *(_QWORD *)v46;
      }
      else
      {
        if ( v23 - 1 == (unsigned int)sub_C444A0(v46) )
          goto LABEL_29;
        v24 = v23 == (unsigned int)sub_C445E0(v46);
      }
      if ( !v24 )
        return 0;
LABEL_29:
      if ( (unsigned __int8)sub_920620(a2) )
      {
        v36 = *a1;
        v54[0] = _mm_loadu_si128(*a1 + 6);
        v54[1] = _mm_loadu_si128(v36 + 7);
        v55 = _mm_loadu_si128(v36 + 8);
        v56 = _mm_loadu_si128(v36 + 9);
        v37 = v36[10].m128i_i64[0];
        v55.m128i_i64[1] = a2;
        v57 = v37;
        v38 = a4[1];
        if ( *(_BYTE *)(v38 + 8) == 17 )
        {
          v39 = *(_DWORD *)(v38 + 32);
          v53 = v39;
          if ( v39 > 0x40 )
          {
            sub_C43690((__int64)&v52, -1, 1);
          }
          else
          {
            v40 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v39;
            if ( !v39 )
              v40 = 0;
            v52 = v40;
          }
        }
        else
        {
          v53 = 1;
          v52 = 1;
        }
        if ( (v48 & 2) != 0 )
          v41 = sub_9B3E70(a4, (__int64 *)&v52, 0, 0, v54) & 0x3FC;
        else
          LOBYTE(v41) = sub_9B3E70(a4, (__int64 *)&v52, 3, 0, v54);
        if ( (v48 & 4) != 0 )
          LOBYTE(v41) = v41 & 0xFB;
        if ( v53 > 0x40 && v52 )
        {
          v47 = v41;
          j_j___libc_free_0_0(v52);
          LOBYTE(v41) = v47;
        }
        if ( (v41 & 3) != 0 )
          return 0;
      }
      v29 = (unsigned int **)(*a1)[2].m128i_i64[0];
      v55.m128i_i16[0] = 257;
      if ( a5 )
      {
        v30 = v13;
        v13 = v51;
        v51 = v30;
      }
      v31 = (unsigned __int8 *)sub_B36550(v29, *(_QWORD *)(a2 - 96), (__int64)v13, (__int64)v51, (__int64)v54, a2);
      if ( (unsigned __int8)sub_920620(a2) )
        sub_B45150((__int64)v31, v48);
      sub_BD6B90(v31, a3);
      v32 = *a3 - 29;
      v55.m128i_i16[0] = 257;
      v25 = (unsigned __int8 *)sub_B504D0(v32, (__int64)a4, (__int64)v31, (__int64)v54, 0, 0);
      sub_B45260(v25, (__int64)a3, 1);
      if ( (unsigned __int8)sub_920620(a2) )
      {
        v33 = 0;
        v34 = (v48 >> 1) & 1;
        if ( !sub_B451C0((__int64)v25) )
          v34 = 0;
        sub_B44EF0((__int64)v25, v34);
        if ( sub_B451D0((__int64)v25) )
          v33 = (v48 >> 2) & 1;
        sub_B44F10((__int64)v25, v33);
        v35 = sub_B451E0((__int64)v25);
        sub_B450D0((__int64)v25, v49 & v35);
      }
      return v25;
    default:
      return 0;
  }
}
