// Function: sub_202D4A0
// Address: 0x202d4a0
//
__int64 *__fastcall sub_202D4A0(__int64 **a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v5; // rsi
  char *v6; // rdx
  char v7; // al
  __int64 v8; // rdx
  unsigned int v9; // eax
  const void **v10; // rdx
  unsigned int v11; // ebx
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rbx
  unsigned __int8 v17; // al
  __int128 v18; // rax
  __int64 *v19; // rax
  __int64 *v20; // rdx
  __int64 *v21; // r9
  __int64 *v22; // r8
  __int64 v23; // rdx
  __int64 **v24; // rdx
  __int64 *v25; // r15
  __int64 (__fastcall *v26)(__int64, __int64); // r14
  __int64 v27; // rax
  unsigned int v28; // edx
  __int64 *v29; // r14
  __int128 v31; // [rsp-10h] [rbp-2C0h]
  unsigned int *i; // [rsp+18h] [rbp-298h]
  const void **v34; // [rsp+20h] [rbp-290h]
  unsigned int v35; // [rsp+28h] [rbp-288h]
  unsigned int v36; // [rsp+28h] [rbp-288h]
  __int64 v37; // [rsp+30h] [rbp-280h]
  unsigned int *v38; // [rsp+38h] [rbp-278h]
  __int64 *v39; // [rsp+40h] [rbp-270h]
  __int64 *v40; // [rsp+40h] [rbp-270h]
  __int64 *v41; // [rsp+48h] [rbp-268h]
  __int64 v42; // [rsp+50h] [rbp-260h] BYREF
  int v43; // [rsp+58h] [rbp-258h]
  char v44[8]; // [rsp+60h] [rbp-250h] BYREF
  __int64 v45; // [rsp+68h] [rbp-248h]
  _BYTE *v46; // [rsp+70h] [rbp-240h] BYREF
  __int64 v47; // [rsp+78h] [rbp-238h]
  _BYTE v48[560]; // [rsp+80h] [rbp-230h] BYREF

  v5 = *(_QWORD *)(a2 + 72);
  v42 = v5;
  if ( v5 )
    sub_1623A60((__int64)&v42, v5, 2);
  v6 = *(char **)(a2 + 40);
  v43 = *(_DWORD *)(a2 + 64);
  v46 = v48;
  v47 = 0x2000000000LL;
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v44[0] = v7;
  v45 = v8;
  if ( v7 )
  {
    switch ( v7 )
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
        LOBYTE(v9) = 2;
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
        LOBYTE(v9) = 3;
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
        LOBYTE(v9) = 4;
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
        LOBYTE(v9) = 5;
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
        LOBYTE(v9) = 6;
        break;
      case 55:
        LOBYTE(v9) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v9) = 8;
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
        LOBYTE(v9) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v9) = 10;
        break;
    }
    v34 = 0;
  }
  else
  {
    LOBYTE(v9) = sub_1F596B0((__int64)v44);
    v35 = v9;
    v34 = v10;
  }
  v11 = v35;
  LOBYTE(v11) = v9;
  v36 = v11;
  v38 = *(unsigned int **)(a2 + 32);
  for ( i = &v38[10 * *(unsigned int *)(a2 + 56)]; i != v38; v38 += 10 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)v38 + 40LL) + 16LL * v38[2];
    v13 = *(_BYTE *)v12;
    v14 = *(_QWORD *)(v12 + 8);
    v44[0] = v13;
    v45 = v14;
    if ( v13 )
      v15 = word_4305480[(unsigned __int8)(v13 - 14)];
    else
      v15 = sub_1F58D30((__int64)v44);
    if ( v15 )
    {
      v16 = 0;
      v37 = v15;
      do
      {
        v25 = a1[1];
        v39 = *a1;
        v26 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
        v27 = sub_1E0A0C0(v25[4]);
        if ( v26 == sub_1D13A20 )
        {
          v28 = 8 * sub_15A9520(v27, 0);
          if ( v28 == 32 )
          {
            v17 = 5;
          }
          else if ( v28 <= 0x20 )
          {
            v17 = 3;
            if ( v28 != 8 )
              v17 = 4 * (v28 == 16);
          }
          else
          {
            v17 = 6;
            if ( v28 != 64 )
            {
              v17 = 0;
              if ( v28 == 128 )
                v17 = 7;
            }
          }
        }
        else
        {
          v17 = v26((__int64)v39, v27);
        }
        *(_QWORD *)&v18 = sub_1D38BB0((__int64)v25, v16, (__int64)&v42, v17, 0, 0, a3, a4, a5, 0);
        v19 = sub_1D332F0(
                v25,
                106,
                (__int64)&v42,
                v36,
                v34,
                0,
                *(double *)a3.m128i_i64,
                a4,
                a5,
                *(_QWORD *)v38,
                *((_QWORD *)v38 + 1),
                v18);
        v21 = v20;
        v22 = v19;
        v23 = (unsigned int)v47;
        if ( (unsigned int)v47 >= HIDWORD(v47) )
        {
          v40 = v19;
          v41 = v21;
          sub_16CD150((__int64)&v46, v48, 0, 16, (int)v19, (int)v21);
          v23 = (unsigned int)v47;
          v22 = v40;
          v21 = v41;
        }
        v24 = (__int64 **)&v46[16 * v23];
        ++v16;
        *v24 = v22;
        v24[1] = v21;
        LODWORD(v47) = v47 + 1;
      }
      while ( v16 != v37 );
    }
  }
  *((_QWORD *)&v31 + 1) = (unsigned int)v47;
  *(_QWORD *)&v31 = v46;
  v29 = sub_1D359D0(
          a1[1],
          104,
          (__int64)&v42,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          *(double *)a3.m128i_i64,
          a4,
          a5,
          v31);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
  return v29;
}
