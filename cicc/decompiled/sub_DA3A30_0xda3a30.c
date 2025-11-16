// Function: sub_DA3A30
// Address: 0xda3a30
//
__int64 *__fastcall sub_DA3A30(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // rcx
  unsigned int v5; // r8d
  unsigned __int8 v6; // al
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // eax
  int v13; // eax
  __int64 v14; // rax
  _BYTE *v16; // r11
  unsigned __int8 *v17; // r13
  char v18; // cl
  unsigned __int8 *v19; // rbx
  char v20; // al
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rcx
  bool v26; // al
  __int64 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int8 *v34; // rbx
  unsigned __int8 *v35; // r14
  __int64 v36; // rax
  __int64 v37; // r13
  unsigned __int8 *v38; // r13
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int8 *v43; // rbx
  __int64 v44; // rax
  __int64 v45; // r14
  _BYTE *v46; // rax
  unsigned __int64 v47; // rbx
  _BYTE *v48; // [rsp+8h] [rbp-A8h]
  _BYTE *v49; // [rsp+8h] [rbp-A8h]
  char v50; // [rsp+16h] [rbp-9Ah]
  char v51; // [rsp+17h] [rbp-99h]
  unsigned int v52; // [rsp+18h] [rbp-98h]
  __int64 v53; // [rsp+18h] [rbp-98h]
  unsigned int v54; // [rsp+18h] [rbp-98h]
  _BYTE *v55; // [rsp+18h] [rbp-98h]
  unsigned int v56; // [rsp+20h] [rbp-90h] BYREF
  __int64 v57; // [rsp+28h] [rbp-88h]
  _BYTE *v58; // [rsp+30h] [rbp-80h]
  char v59; // [rsp+38h] [rbp-78h]
  char v60; // [rsp+39h] [rbp-77h]
  unsigned __int8 *v61; // [rsp+40h] [rbp-70h]
  char v62; // [rsp+48h] [rbp-68h]
  unsigned int v63; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int8 *v64; // [rsp+58h] [rbp-58h]
  _BYTE *v65; // [rsp+60h] [rbp-50h]
  char v66; // [rsp+68h] [rbp-48h]
  char v67; // [rsp+69h] [rbp-47h]
  unsigned __int8 *v68; // [rsp+70h] [rbp-40h]
  char v69; // [rsp+78h] [rbp-38h]

  if ( !sub_D97040(a1, *((_QWORD *)a2 + 1)) )
    return sub_DA3860((_QWORD *)a1, (__int64)a2);
  v6 = *a2;
  if ( *a2 > 0x1Cu )
  {
    v7 = *((_QWORD *)a2 + 5);
    v8 = *(_QWORD *)(a1 + 40);
    if ( v7 )
    {
      v4 = (unsigned int)(*(_DWORD *)(v7 + 44) + 1);
      v9 = *(_DWORD *)(v7 + 44) + 1;
    }
    else
    {
      v4 = 0;
      v9 = 0;
    }
    if ( v9 < *(_DWORD *)(v8 + 32) && *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v4) )
      goto LABEL_7;
    a2 = (unsigned __int8 *)sub_ACADE0(*((__int64 ***)a2 + 1));
    return sub_DA3860((_QWORD *)a1, (__int64)a2);
  }
  if ( v6 == 17 )
    return sub_DA2570(a1, (__int64)a2);
  if ( v6 != 5 )
    return sub_DA3860((_QWORD *)a1, (__int64)a2);
  v8 = *(_QWORD *)(a1 + 40);
LABEL_7:
  sub_D94080((__int64)&v56, a2, (__int64 *)v8, v4, v5);
  if ( v62 )
  {
    v16 = v58;
    v17 = (unsigned __int8 *)v57;
    v52 = v56;
    v18 = *v58;
    switch ( v56 )
    {
      case 0xDu:
      case 0x11u:
        v19 = v61;
        v51 = v59;
        v50 = v60;
        break;
      case 0xFu:
      case 0x13u:
      case 0x16u:
        goto LABEL_18;
      case 0x19u:
      case 0x1Bu:
      case 0x1Eu:
        if ( v18 == 17 )
          goto LABEL_18;
        return 0;
      case 0x1Au:
        return sub_DA3860((_QWORD *)a1, (__int64)a2);
      case 0x1Cu:
      case 0x1Du:
        if ( v18 != 17 )
        {
          v55 = v58;
          v26 = sub_BCAC40(*(_QWORD *)(v57 + 8), 1);
          v16 = v55;
          if ( !v26 )
            return 0;
        }
LABEL_18:
        v53 = (__int64)v16;
        sub_94F890(a3, (__int64)v17);
        sub_94F890(a3, v53);
        return 0;
      default:
        BUG();
    }
    while ( 1 )
    {
      if ( a2 != v19 )
      {
        if ( v19 )
        {
          v48 = v16;
          v22 = sub_D98300(a1, (__int64)v19);
          v16 = v48;
          if ( v22 )
            break;
        }
      }
      v23 = *(unsigned int *)(a3 + 8);
      v24 = *(unsigned int *)(a3 + 12);
      if ( v23 + 1 > v24 )
      {
        v49 = v16;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v23 + 1, 8u, v10, v11);
        v23 = *(unsigned int *)(a3 + 8);
        v16 = v49;
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v23) = v16;
      ++*(_DWORD *)(a3 + 8);
      sub_D94080((__int64)&v63, v17, *(__int64 **)(a1 + 40), v24, v10);
      v21 = v63;
      if ( !v69 )
        goto LABEL_42;
      if ( v52 == 13 )
      {
        if ( (v63 & 0xFFFFFFFD) != 0xD )
          goto LABEL_42;
      }
      else if ( v63 != 17 && v52 == 17 )
      {
        goto LABEL_42;
      }
      if ( v19 && (v51 || v50) && *v19 > 0x1Cu )
      {
        v54 = v63;
        v20 = sub_98F650((__int64)v19, (__int64)v17, (unsigned __int8 *)v63, v25, v10);
        v21 = v54;
        if ( v20 )
        {
LABEL_42:
          sub_94F890(a3, (__int64)v17);
          return 0;
        }
      }
      v17 = v64;
      v52 = v21;
      v16 = v65;
      v19 = v68;
      v51 = v66;
      v50 = v67;
    }
    sub_94F890(a3, (__int64)v19);
    return 0;
  }
  v12 = *a2;
  if ( (unsigned __int8)v12 <= 0x1Cu )
    v13 = *((unsigned __int16 *)a2 + 1);
  else
    v13 = v12 - 29;
  switch ( v13 )
  {
    case 5:
    case 56:
      v28 = sub_B494D0((__int64)a2, 52);
      if ( v28 )
      {
        sub_94F890(a3, v28);
      }
      else if ( *a2 == 85 )
      {
        v29 = *((_QWORD *)a2 - 4);
        if ( v29 )
        {
          if ( !*(_BYTE *)v29 && *(_QWORD *)(v29 + 24) == *((_QWORD *)a2 + 10) && (*(_BYTE *)(v29 + 33) & 0x20) != 0 )
          {
            v30 = *(_DWORD *)(v29 + 36);
            if ( v30 == 292 )
              goto LABEL_56;
            if ( v30 > 0x124 )
            {
              switch ( v30 )
              {
                case 0x149u:
                case 0x14Au:
                case 0x167u:
                case 0x16Du:
                case 0x16Eu:
                case 0x173u:
                  sub_94F890(a3, *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]);
                  sub_94F890(a3, *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
                  break;
                case 0x158u:
                  goto LABEL_56;
                default:
                  return 0;
              }
              return 0;
            }
            if ( v30 == 1 || v30 == 7 )
LABEL_56:
              sub_94F890(a3, *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]);
          }
        }
      }
      return 0;
    case 20:
    case 23:
      v31 = (__int64 *)sub_986520((__int64)a2);
      sub_94F890(a3, *v31);
      v32 = sub_986520((__int64)a2);
      sub_94F890(a3, *(_QWORD *)(v32 + 32));
      return 0;
    case 34:
      v33 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
      if ( (a2[7] & 0x40) != 0 )
      {
        v34 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        v35 = &v34[v33];
      }
      else
      {
        v35 = a2;
        v34 = &a2[-v33];
      }
      if ( v35 != v34 )
      {
        v36 = *(unsigned int *)(a3 + 8);
        do
        {
          v37 = *(_QWORD *)v34;
          if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v36 + 1, 8u, v10, v11);
            v36 = *(unsigned int *)(a3 + 8);
          }
          v34 += 32;
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v36) = v37;
          v36 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
          *(_DWORD *)(a3 + 8) = v36;
        }
        while ( v35 != v34 );
      }
      return 0;
    case 38:
    case 39:
    case 40:
    case 47:
      goto LABEL_46;
    case 48:
      return sub_DA3860((_QWORD *)a1, (__int64)a2);
    case 49:
      if ( !sub_D97040(a1, *((_QWORD *)a2 + 1)) )
        return sub_DA3860((_QWORD *)a1, (__int64)a2);
      v14 = sub_986520((__int64)a2);
      if ( !sub_D97040(a1, *(_QWORD *)(*(_QWORD *)v14 + 8LL)) )
        return sub_DA3860((_QWORD *)a1, (__int64)a2);
LABEL_46:
      v27 = (__int64 *)sub_986520((__int64)a2);
      sub_94F890(a3, *v27);
      return 0;
    case 57:
      v38 = a2;
      if ( sub_BCAC40(*((_QWORD *)a2 + 1), 1) )
        goto LABEL_68;
      v41 = *(_QWORD *)sub_986520((__int64)a2);
      if ( *(_BYTE *)v41 != 82 )
        goto LABEL_68;
      if ( (unsigned __int16)((*(_WORD *)(v41 + 2) & 0x3F) - 32) > 1u )
      {
        v47 = sub_D97050(a1, *(_QWORD *)(*(_QWORD *)(v41 - 64) + 8LL));
        if ( v47 > sub_D97050(a1, *((_QWORD *)a2 + 1)) )
          return sub_DA3860((_QWORD *)a1, (__int64)a2);
      }
      else
      {
        v46 = *(_BYTE **)(v41 - 32);
        if ( *v46 != 17 || !sub_9867B0((__int64)(v46 + 24)) )
          return sub_DA3860((_QWORD *)a1, (__int64)a2);
      }
LABEL_68:
      v42 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
      v43 = &a2[-v42];
      if ( (a2[7] & 0x40) != 0 )
      {
        v43 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        v38 = &v43[v42];
      }
      if ( v38 != v43 )
      {
        v44 = *(unsigned int *)(a3 + 8);
        do
        {
          v45 = *(_QWORD *)v43;
          if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v44 + 1, 8u, v39, v40);
            v44 = *(unsigned int *)(a3 + 8);
          }
          v43 += 32;
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v44) = v45;
          v44 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
          *(_DWORD *)(a3 + 8) = v44;
        }
        while ( v38 != v43 );
      }
      return 0;
    default:
      return 0;
  }
}
