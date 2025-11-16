// Function: sub_1F877C0
// Address: 0x1f877c0
//
__int64 *__fastcall sub_1F877C0(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rax
  char v13; // di
  const void **v14; // rdx
  __int64 v15; // rdx
  int v16; // ecx
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // ecx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rdi
  int v25; // eax
  bool v26; // al
  __int64 *v27; // r13
  __int64 v29; // rdx
  int v30; // ecx
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  __int64 v36; // rdi
  unsigned __int64 v37; // rax
  int v38; // r14d
  int v39; // eax
  unsigned __int64 v40; // rax
  __int64 *v41; // r12
  __int128 v42; // rax
  __int128 v43; // rax
  __int64 v44; // r12
  unsigned __int64 v45; // rax
  char *v46; // rsi
  __int64 v47; // rcx
  unsigned __int8 v48; // dl
  const void **v49; // r15
  unsigned int v50; // r13d
  __int64 *v51; // r12
  unsigned __int64 v52; // rax
  __int128 v53; // rax
  const void **v54; // rdx
  __int64 v55; // [rsp+18h] [rbp-A8h]
  __int64 v56; // [rsp+28h] [rbp-98h]
  unsigned int v57; // [rsp+30h] [rbp-90h]
  const void **v58; // [rsp+30h] [rbp-90h]
  unsigned int v59; // [rsp+30h] [rbp-90h]
  unsigned int v60; // [rsp+3Ch] [rbp-84h]
  unsigned int v62; // [rsp+48h] [rbp-78h]
  __int64 v63; // [rsp+50h] [rbp-70h]
  __int64 v64; // [rsp+60h] [rbp-60h] BYREF
  int v65; // [rsp+68h] [rbp-58h]
  unsigned int v66; // [rsp+70h] [rbp-50h] BYREF
  const void **v67; // [rsp+78h] [rbp-48h]
  char v68[8]; // [rsp+80h] [rbp-40h] BYREF
  const void **v69; // [rsp+88h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 72);
  v64 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v64, v6, 2);
  v65 = *(_DWORD *)(a2 + 64);
  v7 = *(__int64 **)(a2 + 32);
  v8 = *v7;
  v9 = v7[1];
  v10 = v7[5];
  v11 = v7[6];
  v63 = *v7;
  v60 = *((_DWORD *)v7 + 12);
  v12 = *(_QWORD *)(a2 + 40);
  v13 = *(_BYTE *)v12;
  v14 = *(const void ***)(v12 + 8);
  LOBYTE(v66) = v13;
  v67 = v14;
  if ( !v13 )
  {
    v58 = v14;
    if ( sub_1F58D20((__int64)&v66) )
    {
      v68[0] = sub_1F596B0((__int64)&v66);
      v13 = v68[0];
      v69 = v54;
      if ( v68[0] )
        goto LABEL_5;
    }
    else
    {
      v68[0] = 0;
      v69 = v58;
    }
    v62 = sub_1F58D40((__int64)v68);
    v19 = sub_1D1ADA0(v10, v11, v29, v30, v31, v32);
    if ( !v19 )
      goto LABEL_17;
    goto LABEL_6;
  }
  if ( (unsigned __int8)(v13 - 14) <= 0x5Fu )
  {
    switch ( v13 )
    {
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
      default:
        v13 = 2;
        break;
    }
  }
LABEL_5:
  v62 = sub_1F6C8D0(v13);
  v19 = sub_1D1ADA0(v10, v11, v15, v16, v17, v18);
  if ( !v19 )
    goto LABEL_17;
LABEL_6:
  v24 = *(_QWORD *)(v19 + 88);
  v20 = *(unsigned int *)(v24 + 32);
  if ( (unsigned int)v20 <= 0x40 )
  {
    v26 = *(_QWORD *)(v24 + 24) == 0;
  }
  else
  {
    v57 = *(_DWORD *)(v24 + 32);
    v25 = sub_16A57B0(v24 + 24);
    v20 = v57;
    v26 = v57 == v25;
  }
  if ( v26 )
  {
    v27 = (__int64 *)v8;
    goto LABEL_10;
  }
LABEL_17:
  v33 = sub_1D1ADA0(v10, v11, v20, v21, v22, v23);
  if ( v33 )
  {
    v34 = *(_QWORD *)(v33 + 88);
    v35 = v62;
    v36 = v34 + 24;
    v59 = *(_DWORD *)(v34 + 32);
    if ( v59 > 0x40 )
    {
      v55 = *(_QWORD *)(v33 + 88);
      v56 = v34 + 24;
      v39 = sub_16A57B0(v36);
      v36 = v56;
      v35 = v62;
      if ( v59 - v39 > 0x40 )
        goto LABEL_26;
      v37 = **(_QWORD **)(v55 + 24);
    }
    else
    {
      v37 = *(_QWORD *)(v34 + 24);
    }
    if ( v35 > v37 )
      goto LABEL_21;
LABEL_26:
    v40 = sub_16A6CD0(v36, v35);
    v41 = (__int64 *)*a1;
    *(_QWORD *)&v42 = sub_1D38BB0(
                        *a1,
                        v40,
                        (__int64)&v64,
                        *(unsigned __int8 *)(*(_QWORD *)(v10 + 40) + 16LL * v60),
                        *(const void ***)(*(_QWORD *)(v10 + 40) + 16LL * v60 + 8),
                        0,
                        a3,
                        a4,
                        a5,
                        0);
    v27 = sub_1D332F0(
            v41,
            *(unsigned __int16 *)(a2 + 24),
            (__int64)&v64,
            v66,
            v67,
            0,
            *(double *)a3.m128i_i64,
            a4,
            a5,
            v8,
            v9,
            v42);
    goto LABEL_10;
  }
LABEL_21:
  if ( *(_WORD *)(v10 + 24) == 145
    && *(_WORD *)(**(_QWORD **)(v10 + 32) + 24LL) == 118
    && (*(_QWORD *)&v43 = sub_1F87630((__int64 **)a1, v10, *(double *)a3.m128i_i64, a4, a5), (_QWORD)v43) )
  {
    v27 = sub_1D332F0(
            (__int64 *)*a1,
            *(unsigned __int16 *)(a2 + 24),
            (__int64)&v64,
            v66,
            v67,
            0,
            *(double *)a3.m128i_i64,
            a4,
            a5,
            v8,
            v9,
            v43);
  }
  else
  {
    v38 = *(unsigned __int16 *)(v63 + 24);
    if ( (unsigned int)(v38 - 125) > 1 )
      goto LABEL_23;
    v44 = sub_1D23600(*a1, v10);
    v45 = sub_1D23600(*a1, *(_QWORD *)(*(_QWORD *)(v63 + 32) + 40LL));
    if ( !v44 )
      goto LABEL_23;
    if ( !v45 )
      goto LABEL_23;
    v46 = *(char **)(v44 + 40);
    v47 = *(_QWORD *)(v45 + 40);
    v48 = *v46;
    if ( *(_BYTE *)v47 != *v46 )
      goto LABEL_23;
    v49 = (const void **)*((_QWORD *)v46 + 1);
    if ( *(const void ***)(v47 + 8) != v49 && !v48 )
      goto LABEL_23;
    v50 = v48;
    v51 = sub_1D32920(
            (_QWORD *)*a1,
            (unsigned int)(v38 != *(unsigned __int16 *)(a2 + 24)) + 52,
            (__int64)&v64,
            v48,
            *((_QWORD *)v46 + 1),
            v44,
            *(double *)a3.m128i_i64,
            a4,
            a5,
            v45);
    if ( v51 )
    {
      v52 = sub_1D38BB0(*a1, v62, (__int64)&v64, v50, v49, 0, a3, a4, a5, 0);
      *(_QWORD *)&v53 = sub_1D32920(
                          (_QWORD *)*a1,
                          0x39u,
                          (__int64)&v64,
                          v50,
                          (__int64)v49,
                          (__int64)v51,
                          *(double *)a3.m128i_i64,
                          a4,
                          a5,
                          v52);
      v27 = sub_1D332F0(
              (__int64 *)*a1,
              *(unsigned __int16 *)(a2 + 24),
              (__int64)&v64,
              v66,
              v67,
              0,
              *(double *)a3.m128i_i64,
              a4,
              a5,
              **(_QWORD **)(v63 + 32),
              *(_QWORD *)(*(_QWORD *)(v63 + 32) + 8LL),
              v53);
    }
    else
    {
LABEL_23:
      v27 = 0;
    }
  }
LABEL_10:
  if ( v64 )
    sub_161E7C0((__int64)&v64, v64);
  return v27;
}
