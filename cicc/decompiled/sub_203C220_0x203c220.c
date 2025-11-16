// Function: sub_203C220
// Address: 0x203c220
//
__int64 *__fastcall sub_203C220(__int64 *a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  int v6; // ecx
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rax
  char v11; // dl
  __int64 v12; // rax
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  unsigned __int64 v15; // r14
  unsigned int v16; // edx
  __int16 *v17; // r15
  __int128 v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // r10
  __int64 v21; // r8
  __int64 *v22; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // [rsp+8h] [rbp-A8h]
  _QWORD *v27; // [rsp+10h] [rbp-A0h]
  __int64 v28; // [rsp+10h] [rbp-A0h]
  unsigned int v29; // [rsp+18h] [rbp-98h]
  __int64 *v30; // [rsp+18h] [rbp-98h]
  int v31; // [rsp+20h] [rbp-90h]
  unsigned int v32; // [rsp+20h] [rbp-90h]
  __int128 v33; // [rsp+20h] [rbp-90h]
  unsigned int v34; // [rsp+40h] [rbp-70h] BYREF
  const void **v35; // [rsp+48h] [rbp-68h]
  _BYTE v36[8]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v37; // [rsp+58h] [rbp-58h]
  __int64 v38; // [rsp+60h] [rbp-50h] BYREF
  int v39; // [rsp+68h] [rbp-48h]
  const void **v40; // [rsp+70h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v38,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v34) = v39;
  v35 = v40;
  if ( (_BYTE)v39 )
    v6 = word_4305480[(unsigned __int8)(v39 - 14)];
  else
    v6 = sub_1F58D30((__int64)&v34);
  v7 = *(unsigned __int64 **)(a2 + 32);
  v8 = *v7;
  v9 = v7[1];
  v10 = *(_QWORD *)(*v7 + 40) + 16LL * *((unsigned int *)v7 + 2);
  v11 = *(_BYTE *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v36[0] = v11;
  v37 = v12;
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
        v13 = 2;
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
        v13 = 3;
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
        v13 = 4;
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
        v13 = 5;
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
        v13 = 6;
        break;
      case 55:
        v13 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v13 = 8;
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
        v13 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v13 = 10;
        break;
    }
    v26 = 0;
  }
  else
  {
    v31 = v6;
    v13 = sub_1F596B0((__int64)v36);
    v6 = v31;
    v26 = v14;
  }
  v29 = v6;
  v27 = *(_QWORD **)(a1[1] + 48);
  v32 = v13;
  if ( !(unsigned __int8)sub_1D15020(v13, v6) )
    sub_1F593D0(v27, v32, v26, v29);
  sub_1F40D10((__int64)&v38, *a1, *(_QWORD *)(a1[1] + 48), v36[0], v37);
  if ( (_BYTE)v38 == 6 )
  {
    v24 = sub_202DEF0((__int64)a1, a2, a3, a4, a5);
    return sub_2030300(a1, v24, v25, v34, v35, 0, (__m128i)a3, a4, a5);
  }
  else
  {
    v15 = sub_20363F0((__int64)a1, v8, v9);
    v17 = (__int16 *)(v16 | v9 & 0xFFFFFFFF00000000LL);
    *(_QWORD *)&v18 = sub_20363F0(
                        (__int64)a1,
                        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                        *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
    v19 = *(_QWORD *)(a2 + 72);
    v20 = (__int64 *)a1[1];
    v33 = v18;
    v21 = *(_QWORD *)(a2 + 32);
    v38 = v19;
    if ( v19 )
    {
      v28 = v21;
      v30 = v20;
      sub_1623A60((__int64)&v38, v19, 2);
      v21 = v28;
      v20 = v30;
    }
    v39 = *(_DWORD *)(a2 + 64);
    v22 = sub_1D3A900(
            v20,
            0x89u,
            (__int64)&v38,
            v34,
            v35,
            0,
            a3,
            a4,
            a5,
            v15,
            v17,
            v33,
            *(_QWORD *)(v21 + 80),
            *(_QWORD *)(v21 + 88));
    if ( v38 )
      sub_161E7C0((__int64)&v38, v38);
  }
  return v22;
}
