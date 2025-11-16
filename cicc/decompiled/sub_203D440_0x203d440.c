// Function: sub_203D440
// Address: 0x203d440
//
__int64 *__fastcall sub_203D440(
        __int64 **a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9)
{
  unsigned int v9; // ebx
  char *v10; // rdx
  char v11; // al
  const void **v12; // rdx
  unsigned int v13; // eax
  const void **v14; // rdx
  unsigned int v15; // ecx
  __int64 v16; // rsi
  unsigned int v17; // r12d
  _BYTE *v18; // rax
  _BYTE *i; // rdx
  unsigned int *v20; // r12
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned int v25; // edx
  __int64 v26; // r13
  char v27; // al
  __int64 v28; // rsi
  __int128 v29; // rax
  _BYTE *v30; // r12
  int v31; // edx
  __int64 *v32; // r12
  __int64 v33; // rax
  unsigned int v34; // edx
  __int64 *v35; // r12
  __int128 v37; // [rsp-10h] [rbp-210h]
  __int64 v38; // [rsp+10h] [rbp-1F0h]
  unsigned __int64 v40; // [rsp+20h] [rbp-1E0h]
  const void **v41; // [rsp+28h] [rbp-1D8h]
  unsigned int v42; // [rsp+30h] [rbp-1D0h]
  unsigned int v43; // [rsp+30h] [rbp-1D0h]
  __int64 v44; // [rsp+40h] [rbp-1C0h]
  unsigned __int64 v45; // [rsp+48h] [rbp-1B8h]
  int v46; // [rsp+50h] [rbp-1B0h]
  int v47; // [rsp+54h] [rbp-1ACh]
  __int64 *v48; // [rsp+58h] [rbp-1A8h]
  __int64 (__fastcall *v49)(__int64, __int64); // [rsp+60h] [rbp-1A0h]
  int v50; // [rsp+6Ch] [rbp-194h]
  __int64 *v51; // [rsp+70h] [rbp-190h]
  __int64 v52; // [rsp+90h] [rbp-170h] BYREF
  const void **v53; // [rsp+98h] [rbp-168h]
  __int64 v54; // [rsp+A0h] [rbp-160h] BYREF
  int v55; // [rsp+A8h] [rbp-158h]
  _BYTE v56[8]; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v57; // [rsp+B8h] [rbp-148h]
  _BYTE *v58; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v59; // [rsp+C8h] [rbp-138h]
  _BYTE v60[304]; // [rsp+D0h] [rbp-130h] BYREF

  v10 = *(char **)(a2 + 40);
  v11 = *v10;
  v12 = (const void **)*((_QWORD *)v10 + 1);
  LOBYTE(v52) = v11;
  v53 = v12;
  if ( v11 )
  {
    switch ( v11 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        LOBYTE(v13) = 2;
        break;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        LOBYTE(v13) = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        LOBYTE(v13) = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        LOBYTE(v13) = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        LOBYTE(v13) = 6;
        break;
      case 55:
        LOBYTE(v13) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v13) = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        LOBYTE(v13) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v13) = 10;
        break;
    }
    v41 = 0;
  }
  else
  {
    LOBYTE(v13) = sub_1F596B0((__int64)&v52);
    v42 = v13;
    v41 = v14;
  }
  v15 = v42;
  LOBYTE(v15) = v13;
  v43 = v15;
  v16 = *(_QWORD *)(a2 + 72);
  v54 = v16;
  if ( v16 )
    sub_1623A60((__int64)&v54, v16, 2);
  v55 = *(_DWORD *)(a2 + 64);
  if ( (_BYTE)v52 )
    v17 = word_4305480[(unsigned __int8)(v52 - 14)];
  else
    v17 = sub_1F58D30((__int64)&v52);
  v18 = v60;
  v58 = v60;
  v59 = 0x1000000000LL;
  if ( v17 > 0x10 )
  {
    sub_16CD150((__int64)&v58, v60, v17, 16, a8, a9);
    v18 = v58;
  }
  LODWORD(v59) = v17;
  for ( i = &v18[16 * v17]; i != v18; v18 += 16 )
  {
    if ( v18 )
    {
      *(_QWORD *)v18 = 0;
      *((_DWORD *)v18 + 2) = 0;
    }
  }
  v20 = *(unsigned int **)(a2 + 32);
  v21 = *(_QWORD *)(*(_QWORD *)v20 + 40LL) + 16LL * v20[2];
  v22 = *(_BYTE *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  v56[0] = v22;
  v57 = v23;
  if ( v22 )
    v46 = word_4305480[(unsigned __int8)(v22 - 14)];
  else
    v46 = sub_1F58D30((__int64)v56);
  v24 = *(unsigned int *)(a2 + 56);
  if ( (_DWORD)v24 )
  {
    v40 = 0;
    v47 = 0;
    v38 = 40 * v24;
    while ( 1 )
    {
      a3 = _mm_loadu_si128((const __m128i *)&v20[v40 / 4]);
      v44 = sub_20363F0((__int64)a1, a3.m128i_u64[0], a3.m128i_i64[1]);
      v45 = v25 | a3.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      if ( v46 )
      {
        v26 = 0;
        do
        {
          v32 = a1[1];
          v48 = *a1;
          v50 = v26 + v47;
          v49 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
          v33 = sub_1E0A0C0(v32[4]);
          if ( v49 == sub_1D13A20 )
          {
            v34 = 8 * sub_15A9520(v33, 0);
            if ( v34 == 32 )
            {
              v27 = 5;
            }
            else if ( v34 <= 0x20 )
            {
              v27 = 3;
              if ( v34 != 8 )
                v27 = 4 * (v34 == 16);
            }
            else
            {
              v27 = 6;
              if ( v34 != 64 )
              {
                v27 = 0;
                if ( v34 == 128 )
                  v27 = 7;
              }
            }
          }
          else
          {
            v27 = v49((__int64)v48, v33);
          }
          LOBYTE(v9) = v27;
          v28 = v26++;
          *(_QWORD *)&v29 = sub_1D38BB0((__int64)v32, v28, (__int64)&v54, v9, 0, 0, a3, a4, a5, 0);
          v51 = sub_1D332F0(v32, 106, (__int64)&v54, v43, v41, 0, *(double *)a3.m128i_i64, a4, a5, v44, v45, v29);
          v30 = &v58[16 * v50];
          *(_QWORD *)v30 = v51;
          *((_DWORD *)v30 + 2) = v31;
        }
        while ( v46 != v26 );
        v47 += v46;
      }
      v40 += 40LL;
      if ( v38 == v40 )
        break;
      v20 = *(unsigned int **)(a2 + 32);
    }
  }
  *((_QWORD *)&v37 + 1) = (unsigned int)v59;
  *(_QWORD *)&v37 = v58;
  v35 = sub_1D359D0(a1[1], 104, (__int64)&v54, v52, v53, 0, *(double *)a3.m128i_i64, a4, a5, v37);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  if ( v54 )
    sub_161E7C0((__int64)&v54, v54);
  return v35;
}
