// Function: sub_2141750
// Address: 0x2141750
//
__int64 __fastcall sub_2141750(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  unsigned int *v7; // rax
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r8
  int v12; // r9d
  unsigned int v13; // r15d
  unsigned __int64 v14; // rbx
  _QWORD *v16; // r14
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  const __m128i *v20; // r9
  unsigned int v21; // edx
  _QWORD *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned int v28; // edx
  unsigned int v29; // edx
  unsigned int v30; // edx
  unsigned int v31; // edx
  unsigned int v32; // edx
  unsigned int v33; // edx
  unsigned int v34; // edx
  unsigned int v35; // edx
  unsigned int v36; // edx
  unsigned int v37; // edx
  unsigned int v38; // edx
  unsigned int v39; // edx
  unsigned int v40; // edx
  __int64 v41; // r8
  __int64 v42; // rdx
  unsigned int v43; // edx
  unsigned int v44; // edx
  unsigned int v45; // edx
  unsigned int v46; // edx
  unsigned int v47; // edx
  __int64 v48; // [rsp-8h] [rbp-190h]

  v7 = (unsigned int *)(*(_QWORD *)(a2 + 32) + 40LL * a3);
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v7[2];
  v9 = sub_2016240(a1, a2, *(_BYTE *)v8, *(_QWORD *)(v8 + 8), 0, 0, 0);
  if ( (_BYTE)v9 )
    return 0;
  v13 = v9;
  v14 = 0;
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0x32:
      v19 = (__int64)sub_213CA90((__int64)a1, a2, a4, a5, a6);
      v14 = v36;
      break;
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x4F:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
    case 0x59:
    case 0x5A:
    case 0x5B:
    case 0x5C:
    case 0x5D:
    case 0x5E:
    case 0x5F:
    case 0x60:
    case 0x61:
    case 0x62:
    case 0x63:
    case 0x64:
    case 0x65:
    case 0x66:
    case 0x67:
    case 0x6C:
    case 0x6E:
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7F:
    case 0x80:
    case 0x81:
    case 0x82:
    case 0x83:
    case 0x84:
    case 0x85:
    case 0x8A:
    case 0x8B:
    case 0x8C:
    case 0x8D:
    case 0x94:
    case 0x95:
    case 0x96:
    case 0x97:
    case 0x98:
    case 0x99:
    case 0x9A:
    case 0x9B:
    case 0x9C:
    case 0x9D:
    case 0x9F:
    case 0xA1:
    case 0xA2:
    case 0xA3:
    case 0xA4:
    case 0xA5:
    case 0xA6:
    case 0xA7:
    case 0xA8:
    case 0xA9:
    case 0xAA:
    case 0xAB:
    case 0xAC:
    case 0xAD:
    case 0xAE:
    case 0xAF:
    case 0xB0:
    case 0xB1:
    case 0xB2:
    case 0xB3:
    case 0xB4:
    case 0xB5:
    case 0xB6:
    case 0xB7:
    case 0xB8:
    case 0xB9:
    case 0xBB:
    case 0xBC:
    case 0xBD:
    case 0xBE:
    case 0xC1:
    case 0xC2:
    case 0xC3:
    case 0xC4:
    case 0xC5:
    case 0xC6:
    case 0xC7:
    case 0xC8:
    case 0xC9:
    case 0xCA:
    case 0xCB:
    case 0xCC:
    case 0xCD:
    case 0xCE:
    case 0xCF:
    case 0xD0:
    case 0xD1:
    case 0xD2:
    case 0xD3:
    case 0xD4:
    case 0xD5:
    case 0xD6:
    case 0xD7:
    case 0xD8:
    case 0xD9:
    case 0xDA:
    case 0xDB:
    case 0xDD:
    case 0xDE:
    case 0xDF:
    case 0xE0:
    case 0xE1:
    case 0xE2:
    case 0xE3:
    case 0xE4:
    case 0xE5:
    case 0xE6:
    case 0xE7:
    case 0xE8:
    case 0xE9:
    case 0xEA:
    case 0xEE:
      v19 = (__int64)sub_2141030(a1, a2, a3, (__m128)a4, a5, a6, v48, v11, v12);
      v14 = v33;
      break;
    case 0x44:
    case 0x45:
      v19 = (__int64)sub_2129560(a1, a2, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      v14 = v21;
      break;
    case 0x68:
      v19 = (__int64)sub_213CCB0((__int64)a1, (__int64 *)a2);
      v14 = v46;
      break;
    case 0x69:
      v19 = (__int64)sub_213CE40(a1, a2, a3, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      v14 = v35;
      break;
    case 0x6A:
      v19 = sub_213EEC0((__int64 **)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      v14 = v47;
      break;
    case 0x6B:
      v19 = (__int64)sub_213F3B0((__int64 **)a1, a2, a4, a5, a6, v10, v48, v11, v12);
      v14 = v40;
      break;
    case 0x6D:
      v19 = sub_213F1D0((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      v14 = v39;
      break;
    case 0x6F:
      v22 = (_QWORD *)a1[1];
      v23 = sub_2138AD0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
      goto LABEL_11;
    case 0x7A:
    case 0x7B:
    case 0x7C:
    case 0x7D:
    case 0x7E:
      v16 = (_QWORD *)a1[1];
      v17 = sub_2139210(
              (__int64)a1,
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
              a4,
              a5,
              a6);
      v19 = (__int64)sub_1D2DF70(
                       v16,
                       (__int64 *)a2,
                       **(_QWORD **)(a2 + 32),
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                       (__int64)v17,
                       v18);
      break;
    case 0x86:
    case 0x87:
      v19 = (__int64)sub_2129220((__int64)a1, a2, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      v14 = v25;
      break;
    case 0x88:
      v19 = (__int64)sub_213C7F0((__int64)a1, (__int64 *)a2, *(double *)a4.m128i_i64, a5, a6);
      v14 = v27;
      break;
    case 0x89:
      v19 = (__int64)sub_213C880((__int64)a1, (__int64 *)a2, *(double *)a4.m128i_i64, a5, a6);
      v14 = v29;
      break;
    case 0x8E:
      v19 = (__int64)sub_213D000((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      v14 = v28;
      break;
    case 0x8F:
      v19 = (__int64)sub_213D690((__int64)a1, a2, a4, a5, a6);
      v14 = v32;
      break;
    case 0x90:
      v19 = sub_213C8F0((__int64)a1, a2, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      v14 = v31;
      break;
    case 0x91:
      v19 = sub_213D5D0((__int64)a1, a2, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      v14 = v30;
      break;
    case 0x92:
      v22 = (_QWORD *)a1[1];
      v23 = (__int64)sub_2139100(
                       (__int64)a1,
                       **(_QWORD **)(a2 + 32),
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                       *(double *)a4.m128i_i64,
                       a5,
                       a6);
      goto LABEL_11;
    case 0x93:
    case 0xA0:
      v22 = (_QWORD *)a1[1];
      v23 = (__int64)sub_2139210(
                       (__int64)a1,
                       **(_QWORD **)(a2 + 32),
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                       a4,
                       a5,
                       a6);
LABEL_11:
      v19 = (__int64)sub_1D2DE40(v22, (__int64 *)a2, v23, v24);
      break;
    case 0x9E:
      v19 = sub_200D7B0(
              (__int64)a1,
              **(_QWORD **)(a2 + 32),
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
              **(unsigned __int8 **)(a2 + 40),
              *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
      v14 = v26;
      break;
    case 0xBA:
      v19 = sub_213D130((__int64)a1, a2);
      v14 = v43;
      break;
    case 0xBF:
      v41 = sub_200E230(
              a1,
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
              1,
              0,
              *(double *)a4.m128i_i64,
              a5,
              *(double *)a6.m128i_i64);
      v19 = (__int64)sub_1D2E2F0(
                       (_QWORD *)a1[1],
                       (__int64 *)a2,
                       **(_QWORD **)(a2 + 32),
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                       v41,
                       v42,
                       *(_OWORD *)(*(_QWORD *)(a2 + 32) + 80LL));
      break;
    case 0xC0:
      v19 = (__int64)sub_213C770((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      v14 = v38;
      break;
    case 0xDC:
      v19 = (__int64)sub_213C9B0((__int64)a1, a2);
      v14 = v37;
      break;
    case 0xEB:
      v19 = (__int64)sub_21293C0((__int64)a1, a2, a3, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      v14 = v45;
      break;
    case 0xEC:
      v19 = sub_213D210((__int64)a1, a2, a3, *(double *)a4.m128i_i64, a5, a6);
      v14 = v44;
      break;
    case 0xED:
      v19 = sub_21413B0(a1, a2, a3, (__m128)a4, a5, a6, v48, v11, v12);
      v14 = v34;
      break;
  }
  if ( !v19 )
  {
    return 0;
  }
  else if ( a2 == v19 )
  {
    return 1;
  }
  else
  {
    sub_2013400((__int64)a1, a2, 0, v19, (__m128i *)v14, v20);
  }
  return v13;
}
