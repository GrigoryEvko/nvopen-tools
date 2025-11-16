// Function: sub_203E430
// Address: 0x203e430
//
__int64 __fastcall sub_203E430(__int64 *a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  unsigned __int64 v6; // r14
  __int16 *v7; // rdx
  __int16 *v8; // r15
  __int128 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r10
  char *v12; // rax
  char v13; // dl
  __int64 v14; // rax
  unsigned __int8 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  const void **v20; // rdx
  int v21; // ecx
  char v22; // al
  const void **v23; // rsi
  __int64 v24; // rsi
  __int64 *v25; // r11
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  int v28; // ecx
  char v29; // al
  unsigned __int8 v30; // al
  __int64 v31; // rdx
  unsigned int v32; // r14d
  const void **v33; // r15
  _QWORD *v34; // rbx
  unsigned int v35; // eax
  __int64 *v36; // r14
  unsigned int v37; // ebx
  __int64 v38; // rax
  unsigned int v39; // edx
  unsigned __int8 v40; // al
  __int128 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r14
  const void **v46; // rdx
  const void **v47; // rdx
  __int64 v48; // [rsp+0h] [rbp-B0h]
  __int64 (__fastcall *v49)(__int64, __int64, __int64, __int64, __int64); // [rsp+8h] [rbp-A8h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+10h] [rbp-A0h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  _QWORD *v53; // [rsp+18h] [rbp-98h]
  __int64 *v54; // [rsp+18h] [rbp-98h]
  __int64 v55; // [rsp+18h] [rbp-98h]
  __int64 v56; // [rsp+18h] [rbp-98h]
  __int128 v57; // [rsp+20h] [rbp-90h]
  __int64 *v58; // [rsp+20h] [rbp-90h]
  unsigned __int64 v59; // [rsp+28h] [rbp-88h]
  __int64 v60; // [rsp+30h] [rbp-80h]
  unsigned int v61; // [rsp+30h] [rbp-80h]
  int v62; // [rsp+30h] [rbp-80h]
  unsigned int v63; // [rsp+30h] [rbp-80h]
  __int64 (__fastcall *v64)(__int64, __int64); // [rsp+30h] [rbp-80h]
  unsigned int v65; // [rsp+38h] [rbp-78h]
  __int64 v66; // [rsp+40h] [rbp-70h] BYREF
  int v67; // [rsp+48h] [rbp-68h]
  unsigned int v68; // [rsp+50h] [rbp-60h] BYREF
  __int64 v69; // [rsp+58h] [rbp-58h]
  unsigned int v70; // [rsp+60h] [rbp-50h] BYREF
  const void **v71; // [rsp+68h] [rbp-48h]
  __int64 v72; // [rsp+70h] [rbp-40h] BYREF
  int v73; // [rsp+78h] [rbp-38h]

  v6 = sub_20363F0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = v7;
  *(_QWORD *)&v9 = sub_20363F0(
                     (__int64)a1,
                     *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                     *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v10 = *(_QWORD *)(a2 + 72);
  v57 = v9;
  v66 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v66, v10, 2);
  v11 = *a1;
  v67 = *(_DWORD *)(a2 + 64);
  v12 = *(char **)(a2 + 40);
  v48 = v11;
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOBYTE(v68) = v13;
  v69 = v14;
  v15 = (unsigned __int8 *)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v8);
  v16 = *((_QWORD *)v15 + 1);
  v17 = *v15;
  v49 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v11 + 264LL);
  v18 = a1[1];
  v50 = v16;
  v52 = v17;
  v60 = *(_QWORD *)(v18 + 48);
  v19 = sub_1E0A0C0(*(_QWORD *)(v18 + 32));
  v70 = v49(v48, v19, v60, v52, v50);
  v71 = v20;
  if ( !(_BYTE)v68 )
  {
    if ( !sub_1F58D20((__int64)&v68) )
      goto LABEL_11;
LABEL_19:
    if ( sub_1F7E0F0((__int64)&v68) != 2 )
      goto LABEL_11;
    goto LABEL_6;
  }
  if ( (unsigned __int8)(v68 - 14) <= 0x5Fu )
    goto LABEL_19;
  if ( (_BYTE)v68 != 2 )
    goto LABEL_11;
LABEL_6:
  if ( (_BYTE)v70 )
    v21 = word_4305480[(unsigned __int8)(v70 - 14)];
  else
    v21 = sub_1F58D30((__int64)&v70);
  v61 = v21;
  v53 = *(_QWORD **)(a1[1] + 48);
  v22 = sub_1D15020(2, v21);
  v23 = 0;
  if ( !v22 )
  {
    v22 = sub_1F593D0(v53, 2, 0, v61);
    v23 = v47;
  }
  LOBYTE(v70) = v22;
  v71 = v23;
LABEL_11:
  v24 = *(_QWORD *)(a2 + 72);
  v25 = (__int64 *)a1[1];
  v26 = *(_QWORD *)(a2 + 32);
  v72 = v24;
  if ( v24 )
  {
    v51 = v26;
    v54 = v25;
    sub_1623A60((__int64)&v72, v24, 2);
    v26 = v51;
    v25 = v54;
  }
  v73 = *(_DWORD *)(a2 + 64);
  v58 = sub_1D3A900(
          v25,
          0x89u,
          (__int64)&v72,
          v70,
          v71,
          0,
          a3,
          a4,
          a5,
          v6,
          v8,
          v57,
          *(_QWORD *)(v26 + 80),
          *(_QWORD *)(v26 + 88));
  v59 = v27;
  if ( v72 )
    sub_161E7C0((__int64)&v72, v72);
  if ( !(_BYTE)v68 )
  {
    v28 = sub_1F58D30((__int64)&v68);
    v29 = v70;
    if ( (_BYTE)v70 )
      goto LABEL_17;
LABEL_22:
    v62 = v28;
    v30 = sub_1F596B0((__int64)&v70);
    v28 = v62;
    v55 = v31;
    goto LABEL_23;
  }
  v28 = word_4305480[(unsigned __int8)(v68 - 14)];
  v29 = v70;
  if ( !(_BYTE)v70 )
    goto LABEL_22;
LABEL_17:
  switch ( v29 )
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
      v30 = 2;
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
      v30 = 3;
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
      v30 = 4;
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
      v30 = 5;
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
      v30 = 6;
      break;
    case 55:
      v30 = 7;
      break;
    case 86:
    case 87:
    case 88:
    case 98:
    case 99:
    case 100:
      v30 = 8;
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
      v30 = 9;
      break;
    case 94:
    case 95:
    case 96:
    case 97:
    case 106:
    case 107:
    case 108:
    case 109:
      v30 = 10;
      break;
  }
  v55 = 0;
LABEL_23:
  v63 = v28;
  v32 = v30;
  v33 = 0;
  v34 = *(_QWORD **)(a1[1] + 48);
  LOBYTE(v35) = sub_1D15020(v30, v28);
  if ( !(_BYTE)v35 )
  {
    v35 = sub_1F593D0(v34, v32, v55, v63);
    v65 = v35;
    v33 = v46;
  }
  v37 = v65;
  v36 = (__int64 *)a1[1];
  LOBYTE(v37) = v35;
  v56 = *a1;
  v64 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
  v38 = sub_1E0A0C0(v36[4]);
  if ( v64 == sub_1D13A20 )
  {
    v39 = 8 * sub_15A9520(v38, 0);
    if ( v39 == 32 )
    {
      v40 = 5;
    }
    else if ( v39 > 0x20 )
    {
      v40 = 6;
      if ( v39 != 64 )
      {
        v40 = 0;
        if ( v39 == 128 )
          v40 = 7;
      }
    }
    else
    {
      v40 = 3;
      if ( v39 != 8 )
        v40 = 4 * (v39 == 16);
    }
  }
  else
  {
    v40 = v64(v56, v38);
  }
  *(_QWORD *)&v41 = sub_1D38BB0((__int64)v36, 0, (__int64)&v66, v40, 0, 0, (__m128i)a3, a4, a5, 0);
  v42 = sub_1D332F0(v36, 109, (__int64)&v66, v37, v33, 0, *(double *)a3.m128_u64, a4, a5, (__int64)v58, v59, v41);
  v44 = sub_200E230(a1, (__int64)v42, v43, v68, v69, *(double *)a3.m128_u64, a4, *(double *)a5.m128i_i64);
  if ( v66 )
    sub_161E7C0((__int64)&v66, v66);
  return v44;
}
