// Function: sub_990F20
// Address: 0x990f20
//
__int64 __fastcall sub_990F20(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v4; // eax
  int v5; // eax
  unsigned __int8 v6; // cl
  int v7; // edx
  unsigned __int8 *v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // rbx
  unsigned __int8 *v17; // r13
  __int64 v18; // rax
  unsigned __int8 *v19; // r13
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int8 *v22; // r13
  __int64 v23; // rdx
  __int64 *v24; // rbx
  __int64 v25; // rax
  _BYTE *v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-70h] BYREF
  __int64 v34; // [rsp+8h] [rbp-68h] BYREF
  __int64 v35; // [rsp+10h] [rbp-60h] BYREF
  __int64 v36; // [rsp+18h] [rbp-58h] BYREF
  __int64 v37; // [rsp+20h] [rbp-50h] BYREF
  __int64 v38; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v39[2]; // [rsp+30h] [rbp-40h] BYREF
  char v40; // [rsp+40h] [rbp-30h]

  v4 = *a2;
  if ( (unsigned __int8)v4 <= 0x1Cu )
    v5 = *((unsigned __int16 *)a2 + 1);
  else
    v5 = v4 - 29;
  v6 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x1Cu )
    v7 = *(unsigned __int16 *)(a3 + 2);
  else
    v7 = v6 - 29;
  if ( v7 != v5 )
    goto LABEL_6;
  switch ( v7 )
  {
    case 13:
    case 30:
      goto LABEL_13;
    case 15:
      if ( (a2[7] & 0x40) != 0 )
        v22 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v22 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v23 = *(_QWORD *)v22;
      if ( (*(_BYTE *)(a3 + 7) & 0x40) == 0 )
      {
        v24 = (__int64 *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
        v25 = *v24;
        if ( v23 != *v24 )
          goto LABEL_39;
LABEL_64:
        v29 = v24[4];
        v30 = *((_QWORD *)v22 + 4);
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v30;
        *(_QWORD *)(a1 + 8) = v29;
        return a1;
      }
      v24 = *(__int64 **)(a3 - 8);
      v25 = *v24;
      if ( v23 == *v24 )
        goto LABEL_64;
LABEL_39:
      if ( v24[4] != *((_QWORD *)v22 + 4) )
        goto LABEL_6;
      *(_QWORD *)a1 = v23;
      *(_QWORD *)(a1 + 8) = v25;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    case 17:
      if ( ((a2[1] & 2) == 0 || (*(_BYTE *)(a3 + 1) & 2) == 0)
        && (((a2[1] >> 1) & 2) == 0 || (*(_BYTE *)(a3 + 1) & 4) == 0) )
      {
        goto LABEL_6;
      }
      v26 = *(_BYTE **)(sub_986520((__int64)a2) + 32);
      if ( *(_BYTE **)(sub_986520(a3) + 32) != v26 || *v26 != 17 || sub_9867B0((__int64)(v26 + 24)) )
        goto LABEL_6;
      goto LABEL_25;
    case 25:
      if ( ((a2[1] & 2) == 0 || (*(_BYTE *)(a3 + 1) & 2) == 0)
        && (((a2[1] >> 1) & 2) == 0 || (*(_BYTE *)(a3 + 1) & 4) == 0) )
      {
        goto LABEL_6;
      }
      goto LABEL_24;
    case 26:
    case 27:
      if ( (a2[1] & 2) == 0 || (*(_BYTE *)(a3 + 1) & 2) == 0 )
        goto LABEL_6;
LABEL_24:
      v15 = sub_986520((__int64)a2);
      if ( *(_QWORD *)(sub_986520(a3) + 32) != *(_QWORD *)(v15 + 32) )
        goto LABEL_6;
LABEL_25:
      if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
        v16 = *(_QWORD **)(a3 - 8);
      else
        v16 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      if ( (a2[7] & 0x40) != 0 )
        v17 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v17 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v18 = *(_QWORD *)v17;
      *(_QWORD *)(a1 + 8) = *v16;
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v18;
      return a1;
    case 29:
      if ( (a2[1] & 2) == 0 || (*(_BYTE *)(a3 + 1) & 2) == 0 )
        goto LABEL_6;
LABEL_13:
      if ( (a2[7] & 0x40) != 0 )
        v9 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v9 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      if ( (unsigned __int8)(v6 - 42) > 0x11u )
        goto LABEL_6;
      v10 = *(_QWORD *)(a3 - 64);
      v11 = *(_QWORD *)(a3 - 32);
      if ( *(_QWORD *)v9 == v10 && v11 )
        goto LABEL_72;
      if ( v11 == *(_QWORD *)v9 && v10 )
      {
        v11 = v10;
LABEL_72:
        v32 = *((_QWORD *)v9 + 4);
        *(_QWORD *)(a1 + 8) = v11;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v32;
        return a1;
      }
      v12 = *(_QWORD *)(sub_986520((__int64)a2) + 32);
      if ( v10 == v12 && v11 || v11 == v12 && (v11 = v10) != 0 )
      {
        v13 = (__int64 *)sub_986520((__int64)a2);
        *(_QWORD *)(a1 + 8) = v11;
        v14 = *v13;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v14;
        return a1;
      }
      goto LABEL_6;
    case 39:
    case 40:
      if ( (a2[7] & 0x40) != 0 )
        v19 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      else
        v19 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v20 = *(_QWORD *)(*(_QWORD *)v19 + 8LL);
      if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
      {
        v21 = **(_QWORD **)(a3 - 8);
        if ( v20 != *(_QWORD *)(v21 + 8) )
          goto LABEL_6;
      }
      else
      {
        v31 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
        v21 = *(_QWORD *)v31;
        if ( v20 != *(_QWORD *)(*(_QWORD *)v31 + 8LL) )
        {
LABEL_6:
          *(_BYTE *)(a1 + 16) = 0;
          return a1;
        }
      }
      *(_QWORD *)a1 = *(_QWORD *)v19;
      *(_QWORD *)(a1 + 8) = v21;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    case 55:
      v33 = 0;
      v27 = *((_QWORD *)a2 + 5);
      v34 = 0;
      v35 = 0;
      v36 = 0;
      v37 = 0;
      v38 = 0;
      if ( *(_QWORD *)(a3 + 40) != v27 )
        goto LABEL_6;
      if ( !(unsigned __int8)sub_990E50((__int64)a2, &v33, &v34, &v35) )
        goto LABEL_6;
      if ( !(unsigned __int8)sub_990E50(a3, &v36, &v37, &v38) )
        goto LABEL_6;
      sub_990F20(v39, v33, v36);
      if ( !v40 || (unsigned __int8 *)v39[0] != a2 || v39[1] != a3 )
        goto LABEL_6;
      v28 = v34;
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v28;
      *(_QWORD *)(a1 + 8) = v37;
      return a1;
    default:
      goto LABEL_6;
  }
}
