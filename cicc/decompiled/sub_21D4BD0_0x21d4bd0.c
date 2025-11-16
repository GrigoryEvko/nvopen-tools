// Function: sub_21D4BD0
// Address: 0x21d4bd0
//
__int64 *__fastcall sub_21D4BD0(__m128i a1, double a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 *v10; // rax
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  char *v13; // rdx
  char v14; // al
  __int64 v15; // rdx
  char v16; // dl
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // r14
  __int128 v20; // rax
  __int64 *v21; // rdx
  __int64 *v22; // r8
  __int64 *v23; // r9
  __int64 v24; // rax
  __int64 **v25; // rax
  __int128 v26; // rdi
  __int64 *v27; // r12
  __int64 v29; // rax
  const void **v30; // rdx
  __int64 v31; // [rsp+18h] [rbp-128h]
  __int64 *v32; // [rsp+20h] [rbp-120h]
  __int64 *v33; // [rsp+28h] [rbp-118h]
  __int64 v35; // [rsp+38h] [rbp-108h]
  const void **v36; // [rsp+40h] [rbp-100h]
  __int64 v37; // [rsp+48h] [rbp-F8h]
  __int64 v38; // [rsp+50h] [rbp-F0h]
  __int64 v39; // [rsp+58h] [rbp-E8h]
  __int64 v40; // [rsp+60h] [rbp-E0h] BYREF
  int v41; // [rsp+68h] [rbp-D8h]
  char v42[8]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v43; // [rsp+78h] [rbp-C8h]
  _BYTE *v44; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+88h] [rbp-B8h]
  _BYTE v46[176]; // [rsp+90h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(a5 + 72);
  v40 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v40, v8, 2);
  v41 = *(_DWORD *)(a5 + 64);
  v44 = v46;
  v45 = 0x800000000LL;
  v9 = *(unsigned int *)(a5 + 56);
  if ( (_DWORD)v9 )
  {
    v35 = 0;
    v31 = 40 * v9;
    while ( 1 )
    {
      v10 = (__int64 *)(*(_QWORD *)(a5 + 32) + v35);
      v11 = v10[1];
      v12 = *v10;
      v13 = *(char **)(v12 + 40);
      v38 = v12;
      v14 = *v13;
      v15 = *((_QWORD *)v13 + 1);
      v42[0] = v14;
      v43 = v15;
      if ( v14 )
      {
        switch ( v14 )
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
            v16 = 2;
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
            v16 = 3;
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
            v16 = 4;
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
            v16 = 5;
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
            v16 = 6;
            break;
          case 55:
            v16 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v16 = 8;
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
            v16 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v16 = 10;
            break;
        }
        v17 = v39;
        v36 = 0;
        LOBYTE(v17) = v16;
        v39 = v17;
      }
      else
      {
        LOBYTE(v29) = sub_1F596B0((__int64)v42);
        v39 = v29;
        v14 = v42[0];
        v36 = v30;
        if ( !v42[0] )
        {
          v18 = sub_1F58D30((__int64)v42);
          goto LABEL_10;
        }
      }
      v18 = (unsigned __int16)word_435D740[(unsigned __int8)(v14 - 14)];
LABEL_10:
      v19 = 0;
      v37 = v18;
      if ( v18 )
      {
        do
        {
          *(_QWORD *)&v20 = sub_1D38E70((__int64)a7, v19, (__int64)&v40, 0, a1, a2, a3);
          v22 = sub_1D332F0(
                  a7,
                  106,
                  (__int64)&v40,
                  (unsigned int)v39,
                  v36,
                  0,
                  *(double *)a1.m128i_i64,
                  a2,
                  a3,
                  v38,
                  v11,
                  v20);
          v23 = v21;
          v24 = (unsigned int)v45;
          if ( (unsigned int)v45 >= HIDWORD(v45) )
          {
            v33 = v21;
            v32 = v22;
            sub_16CD150((__int64)&v44, v46, 0, 16, (int)v22, (int)v21);
            v24 = (unsigned int)v45;
            v22 = v32;
            v23 = v33;
          }
          v25 = (__int64 **)&v44[16 * v24];
          ++v19;
          *v25 = v22;
          v25[1] = v23;
          LODWORD(v45) = v45 + 1;
        }
        while ( v37 != v19 );
      }
      v35 += 40;
      if ( v31 == v35 )
      {
        *(_QWORD *)&v26 = v44;
        *((_QWORD *)&v26 + 1) = (unsigned int)v45;
        goto LABEL_16;
      }
    }
  }
  *(_QWORD *)&v26 = v46;
  *((_QWORD *)&v26 + 1) = 0;
LABEL_16:
  v27 = sub_1D359D0(
          a7,
          104,
          (__int64)&v40,
          **(unsigned __int8 **)(a5 + 40),
          *(const void ***)(*(_QWORD *)(a5 + 40) + 8LL),
          0,
          *(double *)a1.m128i_i64,
          a2,
          a3,
          v26);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  return v27;
}
