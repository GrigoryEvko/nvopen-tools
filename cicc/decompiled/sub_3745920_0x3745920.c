// Function: sub_3745920
// Address: 0x3745920
//
char __fastcall sub_3745920(__int64 *a1, __int64 a2, int a3)
{
  __int64 v3; // rbp
  __int64 v4; // r12
  __int64 v5; // r15
  char result; // al
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int16 v11; // ax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int16 v14; // bx
  __int64 v15; // rdx
  __int16 v16; // r14
  _QWORD *v17; // rdx
  unsigned int v18; // edx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // rbx
  char v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 *v28; // rdx
  char v29; // al
  __int64 (*v30)(); // r9
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rsi
  int v34; // eax
  int v35; // ecx
  unsigned int v36; // eax
  __int64 v37; // rdx
  int v38; // edi
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rdx
  char v43; // [rsp-A1h] [rbp-A1h]
  unsigned __int64 v44; // [rsp-A0h] [rbp-A0h]
  int v45; // [rsp-98h] [rbp-98h] BYREF
  __int64 v46; // [rsp-90h] [rbp-90h]
  unsigned __int64 v47; // [rsp-88h] [rbp-88h]
  __int64 v48; // [rsp-80h] [rbp-80h]
  __int64 v49; // [rsp-78h] [rbp-78h]
  __int64 v50; // [rsp-70h] [rbp-70h]
  __int16 v51; // [rsp-68h] [rbp-68h] BYREF
  __int64 v52; // [rsp-60h] [rbp-60h]
  unsigned __int64 v53; // [rsp-58h] [rbp-58h]
  __int64 v54; // [rsp-50h] [rbp-50h]
  __int64 v55; // [rsp-48h] [rbp-48h] BYREF
  __int64 v56; // [rsp-40h] [rbp-40h]
  __int64 v57; // [rsp-28h] [rbp-28h]
  __int64 v58; // [rsp-10h] [rbp-10h]
  __int64 v59; // [rsp-8h] [rbp-8h]

  v59 = v3;
  v58 = v5;
  v57 = v4;
  switch ( a3 )
  {
    case 0:
    case 1:
    case 3:
    case 4:
    case 5:
    case 6:
    case 8:
    case 9:
    case 10:
    case 11:
    case 32:
    case 33:
    case 35:
    case 36:
    case 37:
    case 41:
    case 43:
    case 45:
    case 46:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    case 63:
    case 65:
    case 66:
      return 0;
    case 2:
      result = 0;
      if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 1 )
      {
        sub_3741830(
          (__int64)a1,
          *(_QWORD *)(*(_QWORD *)(a1[5] + 56) + 8LL * *(unsigned int *)(*(_QWORD *)(a2 - 32) + 44LL)),
          a2 + 48);
        return 1;
      }
      return result;
    case 7:
      v29 = *(_BYTE *)(a1[13] + 877);
      if ( (v29 & 2) == 0 )
        return 1;
      if ( (v29 & 4) != 0 && *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) != a2 + 24 )
      {
        v40 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v40 )
        {
          if ( *(_BYTE *)(v40 - 24) == 85 )
          {
            v41 = v40 - 24;
            if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v40 + 48), 36) || (unsigned __int8)sub_B49560(v41, 36) )
              return 1;
          }
        }
      }
      v30 = *(__int64 (**)())(*a1 + 56);
      result = 0;
      if ( v30 != sub_3740ED0 )
        return ((unsigned int (__fastcall *)(__int64 *, __int64, __int64, __int64))v30)(a1, 1, 1, 331) != 0;
      return result;
    case 12:
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v28 = *(__int64 **)(a2 - 8);
      else
        v28 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      return sub_3745550(a1, a2, *v28);
    case 13:
      v27 = 56;
      return sub_3749F50(a1, a2, v27);
    case 14:
      v27 = 96;
      return sub_3749F50(a1, a2, v27);
    case 15:
      v27 = 57;
      return sub_3749F50(a1, a2, v27);
    case 16:
      v27 = 97;
      return sub_3749F50(a1, a2, v27);
    case 17:
      v27 = 58;
      return sub_3749F50(a1, a2, v27);
    case 18:
      v27 = 98;
      return sub_3749F50(a1, a2, v27);
    case 19:
      v27 = 60;
      return sub_3749F50(a1, a2, v27);
    case 20:
      v27 = 59;
      return sub_3749F50(a1, a2, v27);
    case 21:
      v27 = 99;
      return sub_3749F50(a1, a2, v27);
    case 22:
      v27 = 62;
      return sub_3749F50(a1, a2, v27);
    case 23:
      v27 = 61;
      return sub_3749F50(a1, a2, v27);
    case 24:
      v27 = 100;
      return sub_3749F50(a1, a2, v27);
    case 25:
      v27 = 190;
      return sub_3749F50(a1, a2, v27);
    case 26:
      v27 = 192;
      return sub_3749F50(a1, a2, v27);
    case 27:
      v27 = 191;
      return sub_3749F50(a1, a2, v27);
    case 28:
      v27 = 186;
      return sub_3749F50(a1, a2, v27);
    case 29:
      v27 = 187;
      return sub_3749F50(a1, a2, v27);
    case 30:
      v27 = 188;
      return sub_3749F50(a1, a2, v27);
    case 31:
      v32 = a1[5];
      v33 = *(_QWORD *)(v32 + 256);
      v34 = *(_DWORD *)(v32 + 272);
      if ( !v34 )
        return 0;
      v35 = v34 - 1;
      v36 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v37 = *(_QWORD *)(v33 + 16LL * v36);
      if ( a2 == v37 )
        return 1;
      v38 = 1;
      while ( 2 )
      {
        if ( v37 == -4096 )
          return 0;
        v36 = v35 & (v38 + v36);
        v37 = *(_QWORD *)(v33 + 16LL * v36);
        if ( a2 != v37 )
        {
          ++v38;
          continue;
        }
        break;
      }
      return 1;
    case 34:
      return sub_374A2E0();
    case 38:
      goto LABEL_30;
    case 39:
      goto LABEL_14;
    case 40:
      v19 = 213;
      return sub_37498B0(a1, a2, v19);
    case 42:
      v19 = 226;
      return sub_37498B0(a1, a2, v19);
    case 44:
      v19 = 220;
      return sub_37498B0(a1, a2, v19);
    case 47:
    case 48:
      v9 = a1[16];
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v10 = *(_QWORD *)(a2 - 8);
      else
        v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v11 = sub_2D5BAE0(v9, a1[14], *(__int64 **)(*(_QWORD *)v10 + 8LL), 0);
      v13 = v12;
      v14 = v11;
      v45 = sub_2D5BAE0(a1[16], a1[14], *(__int64 **)(a2 + 8), 0);
      v16 = v45;
      v46 = v15;
      if ( v14 == (_WORD)v45 )
      {
        if ( v14 || v13 == v15 )
          goto LABEL_8;
        v56 = v13;
        LOWORD(v55) = 0;
        goto LABEL_17;
      }
      LOWORD(v55) = v14;
      v56 = v13;
      if ( !v14 )
      {
LABEL_17:
        v49 = sub_3007260((__int64)&v55);
        v50 = v20;
        goto LABEL_18;
      }
      v49 = sub_3368600(&v55);
      v50 = v39;
LABEL_18:
      v44 = v49;
      v43 = v50;
      if ( (_WORD)v45 )
        v21 = sub_3368600(&v45);
      else
        v21 = sub_3007260((__int64)&v45);
      v47 = v21;
      v48 = v22;
      if ( ((_BYTE)v22 || !v43) && v44 < v47 )
      {
LABEL_14:
        v19 = 214;
        return sub_37498B0(a1, a2, v19);
      }
      if ( v14 == v16 )
      {
        if ( v16 || v13 == v46 )
          goto LABEL_8;
        v52 = v13;
        v51 = 0;
        goto LABEL_24;
      }
      v51 = v14;
      v52 = v13;
      if ( !v14 )
      {
LABEL_24:
        v55 = sub_3007260((__int64)&v51);
        v56 = v23;
        goto LABEL_25;
      }
      v55 = sub_3368600(&v51);
      v56 = v42;
LABEL_25:
      v24 = v55;
      v25 = v56;
      if ( v16 )
        v53 = sub_3368600(&v45);
      else
        v53 = sub_3007260((__int64)&v45);
      v54 = v26;
      if ( (!(_BYTE)v26 || v25) && v53 < v24 )
      {
LABEL_30:
        v19 = 216;
        return sub_37498B0(a1, a2, v19);
      }
LABEL_8:
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v17 = *(_QWORD **)(a2 - 8);
      else
        v17 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v18 = sub_3746830(a1, *v17);
      result = 0;
      if ( v18 )
      {
        sub_3742B00((__int64)a1, (_BYTE *)a2, v18, 1);
        return 1;
      }
      return result;
    case 49:
      return sub_3749A00();
    case 55:
      BUG();
    case 56:
      if ( *(_DWORD *)(a1[13] + 556) != 19 )
        return sub_3749610(a1, a2);
      result = 0;
      if ( *(_BYTE *)a2 == 85 )
      {
        v31 = *(_QWORD *)(a2 - 32);
        if ( v31 )
        {
          if ( !*(_BYTE *)v31 && *(_QWORD *)(v31 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v31 + 33) & 0x20) != 0 )
            return sub_3749610(a1, a2);
        }
      }
      return result;
    case 64:
      return sub_3745530((__int64)a1, a2);
    case 67:
      return sub_3749B60();
    default:
      return 0;
  }
}
