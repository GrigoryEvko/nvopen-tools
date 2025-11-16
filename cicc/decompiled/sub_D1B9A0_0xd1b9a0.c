// Function: sub_D1B9A0
// Address: 0xd1b9a0
//
__int64 __fastcall sub_D1B9A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // r15
  __int64 v9; // r14
  __int64 v10; // rbx
  unsigned __int8 *v11; // r10
  int v12; // eax
  __int16 v13; // ax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int8 *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rsi
  _QWORD *v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rdi
  char v37; // al
  unsigned __int8 *v38; // r10
  int v39; // edx
  int v40; // edx
  unsigned int v41; // eax
  __int64 v42; // rax
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // r10
  __int64 v49; // rax
  __int64 v50; // rdx
  __int16 v51; // ax
  unsigned __int8 *v52; // r10
  __int64 v53; // rsi
  _QWORD *v54; // rax
  _QWORD *v55; // rax
  unsigned int v56; // eax
  __int64 v57; // [rsp+0h] [rbp-50h]
  __int64 v58; // [rsp+0h] [rbp-50h]
  __int64 v59; // [rsp+0h] [rbp-50h]
  __int64 v60; // [rsp+8h] [rbp-48h]
  int v61; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v62; // [rsp+8h] [rbp-48h]
  __int64 v63; // [rsp+8h] [rbp-48h]
  int v64; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v65; // [rsp+8h] [rbp-48h]
  __int64 v66; // [rsp+10h] [rbp-40h]
  unsigned __int8 *v67; // [rsp+10h] [rbp-40h]
  __int64 v68; // [rsp+10h] [rbp-40h]
  __int64 v69; // [rsp+10h] [rbp-40h]
  __int64 v70; // [rsp+10h] [rbp-40h]
  unsigned __int8 *v71; // [rsp+10h] [rbp-40h]

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 14 )
    return 1;
  v6 = *(unsigned __int8 **)(a2 + 16);
  v9 = a3;
  v10 = a4;
  if ( v6 )
  {
    while ( 1 )
    {
      v11 = (unsigned __int8 *)*((_QWORD *)v6 + 3);
      v12 = *v11;
      if ( (unsigned __int8)v12 <= 0x1Cu )
      {
        if ( (_BYTE)v12 == 5 )
        {
          v13 = *((_WORD *)v11 + 1);
          if ( v13 == 34 )
            goto LABEL_13;
          if ( (unsigned __int16)(v13 - 49) <= 1u )
            goto LABEL_44;
        }
        else if ( (unsigned __int8)v12 > 0x15u || (unsigned __int8)v12 <= 3u )
        {
          return 1;
        }
        if ( (unsigned __int8)sub_AC2D20(*((_QWORD *)v6 + 3)) )
          return 1;
        goto LABEL_15;
      }
      if ( (_BYTE)v12 == 61 )
      {
        if ( v9 )
        {
          v31 = *(_QWORD *)(*((_QWORD *)v11 + 5) + 72LL);
          if ( !*(_BYTE *)(v9 + 28) )
            goto LABEL_79;
          v32 = *(_QWORD **)(v9 + 8);
          a4 = *(unsigned int *)(v9 + 20);
          a3 = (__int64)&v32[a4];
          if ( v32 != (_QWORD *)a3 )
          {
            while ( v31 != *v32 )
            {
              if ( (_QWORD *)a3 == ++v32 )
                goto LABEL_50;
            }
            goto LABEL_15;
          }
LABEL_50:
          if ( (unsigned int)a4 < *(_DWORD *)(v9 + 16) )
          {
            a4 = (unsigned int)(a4 + 1);
            *(_DWORD *)(v9 + 20) = a4;
            *(_QWORD *)a3 = v31;
            ++*(_QWORD *)v9;
          }
          else
          {
LABEL_79:
            sub_C8CC70(v9, v31, a3, a4, a5, a6);
          }
        }
        goto LABEL_15;
      }
      if ( (_BYTE)v12 == 62 )
      {
        v33 = *((_QWORD *)v11 - 4);
        if ( !v33 || a2 != v33 )
        {
          if ( a5 != v33 )
            return 1;
          goto LABEL_15;
        }
        if ( !v10 )
          goto LABEL_15;
        v29 = *(_QWORD *)(*((_QWORD *)v11 + 5) + 72LL);
        if ( !*(_BYTE *)(v10 + 28) )
          goto LABEL_84;
        v34 = *(_QWORD **)(v10 + 8);
        a4 = *(unsigned int *)(v10 + 20);
        a3 = (__int64)&v34[a4];
        if ( v34 != (_QWORD *)a3 )
        {
          while ( v29 != *v34 )
          {
            if ( (_QWORD *)a3 == ++v34 )
              goto LABEL_42;
          }
          goto LABEL_15;
        }
        goto LABEL_42;
      }
      a4 = (unsigned int)(unsigned __int8)v12 - 29;
      if ( (unsigned __int8)v12 == 63 )
        goto LABEL_13;
      if ( (unsigned __int8)(v12 - 78) <= 1u )
      {
LABEL_44:
        v16 = a5;
LABEL_14:
        if ( (unsigned __int8)sub_D1B9A0(a1, *((_QWORD *)v6 + 3), v9, v10, v16) )
          return 1;
        goto LABEL_15;
      }
      a3 = (unsigned int)(v12 - 34);
      if ( (unsigned __int8)(v12 - 34) > 0x33u || (v17 = 0x8000000000041LL, !_bittest64(&v17, a3)) )
      {
        if ( (_BYTE)v12 != 82 || **((_BYTE **)v11 - 4) != 20 )
          return 1;
        goto LABEL_15;
      }
      v18 = *((_DWORD *)v11 + 1) & 0x7FFFFFF;
      if ( (_BYTE)v12 == 85 )
        break;
      a3 = 32 * v18;
      if ( v6 < &v11[-a3] )
        goto LABEL_15;
      if ( (unsigned __int8)v12 == 34 )
      {
        if ( v6 >= v11 - 96 )
          goto LABEL_15;
LABEL_27:
        v66 = -96;
        goto LABEL_28;
      }
      if ( (unsigned int)a4 <= 4 )
        goto LABEL_131;
      if ( (unsigned __int8)v12 != 40 )
      {
        if ( (unsigned __int8)v12 != 85 )
          goto LABEL_131;
LABEL_81:
        if ( v6 >= v11 - 32 )
          goto LABEL_15;
LABEL_82:
        v40 = *v11;
        switch ( v40 )
        {
          case '(':
            v62 = v11;
            v41 = sub_B491D0((__int64)v11);
            v11 = v62;
            v66 = -32 - 32LL * v41;
            break;
          case 'U':
            v66 = -32;
            break;
          case '"':
            goto LABEL_27;
          default:
LABEL_131:
            BUG();
        }
LABEL_28:
        if ( (v11[7] & 0x80u) != 0 )
        {
          v57 = (__int64)v11;
          v19 = sub_BD2BC0((__int64)v11);
          v11 = (unsigned __int8 *)v57;
          v60 = v20 + v19;
          if ( *(char *)(v57 + 7) >= 0 )
          {
            if ( (unsigned int)(v60 >> 4) )
LABEL_134:
              BUG();
          }
          else
          {
            v21 = sub_BD2BC0(v57);
            v11 = (unsigned __int8 *)v57;
            if ( (unsigned int)((v60 - v21) >> 4) )
            {
              if ( *(char *)(v57 + 7) >= 0 )
                goto LABEL_134;
              v61 = *(_DWORD *)(sub_BD2BC0(v57) + 8);
              if ( *(char *)(v57 + 7) >= 0 )
                BUG();
              v22 = sub_BD2BC0(v57);
              v11 = (unsigned __int8 *)v57;
              v66 -= 32LL * (unsigned int)(*(_DWORD *)(v22 + v23 - 4) - v61);
            }
          }
        }
        if ( v6 < &v11[v66] )
        {
          v24 = v11;
          v67 = v11;
          v25 = sub_B43CB0((__int64)v11);
          if ( !*(_QWORD *)(a1 + 24) )
            sub_4263D6(v24, v25, v26);
          v27 = (*(__int64 (__fastcall **)(__int64, __int64))(a1 + 32))(a1 + 8, v25);
          v28 = sub_D5D560(v67, v27);
          v11 = v67;
          if ( *(_QWORD *)v6 == v28 )
          {
            if ( !v10 )
              goto LABEL_15;
            v29 = *(_QWORD *)(*((_QWORD *)v67 + 5) + 72LL);
            if ( !*(_BYTE *)(v10 + 28) )
              goto LABEL_84;
            v30 = *(_QWORD **)(v10 + 8);
            a4 = *(unsigned int *)(v10 + 20);
            a3 = (__int64)&v30[a4];
            if ( v30 != (_QWORD *)a3 )
            {
              while ( v29 != *v30 )
              {
                if ( (_QWORD *)a3 == ++v30 )
                  goto LABEL_42;
              }
              goto LABEL_15;
            }
            goto LABEL_42;
          }
        }
        goto LABEL_69;
      }
      v68 = *((_QWORD *)v6 + 3);
      v35 = sub_B491D0(v68);
      v11 = (unsigned __int8 *)v68;
      a3 = v68 - 32LL * v35;
      if ( (unsigned __int64)v6 >= a3 - 32 )
        goto LABEL_15;
      if ( (unsigned __int64)v6 >= v68 - 32 * (unsigned __int64)(*(_DWORD *)(v68 + 4) & 0x7FFFFFF) )
        goto LABEL_82;
LABEL_69:
      v36 = *((_QWORD *)v11 - 4);
      if ( !v36 )
        return 1;
      if ( *(_BYTE *)v36 )
        return 1;
      if ( *(_QWORD *)(v36 + 24) != *((_QWORD *)v11 + 10) )
        return 1;
      v69 = (__int64)v11;
      if ( !sub_B2FC80(v36) )
        return 1;
      v37 = sub_A73ED0((_QWORD *)(v69 + 72), 24);
      v38 = (unsigned __int8 *)v69;
      if ( !v37 )
      {
        v43 = sub_B49560(v69, 24);
        v38 = (unsigned __int8 *)v69;
        if ( !v43 )
          return 1;
      }
      if ( v6 < &v38[-32 * (*((_DWORD *)v38 + 1) & 0x7FFFFFF)] )
        return 1;
      v39 = *v38;
      if ( v39 == 40 )
      {
        v65 = v38;
        v56 = sub_B491D0((__int64)v38);
        v38 = v65;
        v70 = -32 - 32LL * v56;
      }
      else
      {
        v70 = -32;
        if ( v39 != 85 )
        {
          if ( v39 != 34 )
            goto LABEL_131;
          v70 = -96;
        }
      }
      if ( (v38[7] & 0x80u) != 0 )
      {
        v63 = (__int64)v38;
        v44 = sub_BD2BC0((__int64)v38);
        v38 = (unsigned __int8 *)v63;
        if ( *(char *)(v63 + 7) >= 0 )
        {
          if ( (unsigned int)((v44 + v45) >> 4) )
LABEL_132:
            BUG();
        }
        else
        {
          v58 = v44 + v45;
          v46 = sub_BD2BC0(v63);
          v38 = (unsigned __int8 *)v63;
          if ( (unsigned int)((v58 - v46) >> 4) )
          {
            if ( *(char *)(v63 + 7) >= 0 )
              goto LABEL_132;
            v59 = v63;
            v47 = sub_BD2BC0(v63);
            v48 = v63;
            v64 = *(_DWORD *)(v47 + 8);
            if ( *(char *)(v48 + 7) >= 0 )
              BUG();
            v49 = sub_BD2BC0(v59);
            v38 = (unsigned __int8 *)v59;
            v70 -= 32LL * (unsigned int)(*(_DWORD *)(v49 + v50 - 4) - v64);
          }
        }
      }
      if ( v6 >= &v38[v70] )
        return 1;
      v71 = v38;
      v51 = sub_B49EE0(v38, (v6 - &v38[-32 * (*((_DWORD *)v38 + 1) & 0x7FFFFFF)]) >> 5);
      a3 = HIBYTE(v51);
      if ( v51 )
        return 1;
      v52 = v71;
      if ( !v9 )
        goto LABEL_113;
      v53 = *(_QWORD *)(*((_QWORD *)v71 + 5) + 72LL);
      if ( !*(_BYTE *)(v9 + 28) )
        goto LABEL_119;
      v54 = *(_QWORD **)(v9 + 8);
      a4 = *(unsigned int *)(v9 + 20);
      a3 = (__int64)&v54[a4];
      if ( v54 != (_QWORD *)a3 )
      {
        while ( v53 != *v54 )
        {
          if ( (_QWORD *)a3 == ++v54 )
            goto LABEL_120;
        }
        goto LABEL_113;
      }
LABEL_120:
      if ( (unsigned int)a4 < *(_DWORD *)(v9 + 16) )
      {
        a4 = (unsigned int)(a4 + 1);
        *(_DWORD *)(v9 + 20) = a4;
        *(_QWORD *)a3 = v53;
        ++*(_QWORD *)v9;
      }
      else
      {
LABEL_119:
        sub_C8CC70(v9, v53, a3, a4, a5, a6);
        v52 = v71;
      }
LABEL_113:
      if ( !v10 )
        goto LABEL_15;
      v29 = *(_QWORD *)(*((_QWORD *)v52 + 5) + 72LL);
      if ( !*(_BYTE *)(v10 + 28) )
      {
LABEL_84:
        sub_C8CC70(v10, v29, a3, a4, a5, a6);
        goto LABEL_15;
      }
      v55 = *(_QWORD **)(v10 + 8);
      a4 = *(unsigned int *)(v10 + 20);
      a3 = (__int64)&v55[a4];
      if ( v55 != (_QWORD *)a3 )
      {
        while ( v29 != *v55 )
        {
          if ( (_QWORD *)a3 == ++v55 )
            goto LABEL_42;
        }
        goto LABEL_15;
      }
LABEL_42:
      if ( (unsigned int)a4 >= *(_DWORD *)(v10 + 16) )
        goto LABEL_84;
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(v10 + 20) = a4;
      *(_QWORD *)a3 = v29;
      ++*(_QWORD *)v10;
LABEL_15:
      v6 = (unsigned __int8 *)*((_QWORD *)v6 + 1);
      if ( !v6 )
        return 0;
    }
    v42 = *((_QWORD *)v11 - 4);
    if ( v42 )
    {
      if ( !*(_BYTE *)v42
        && *(_QWORD *)(v42 + 24) == *((_QWORD *)v11 + 10)
        && (*(_BYTE *)(v42 + 33) & 0x20) != 0
        && *(_DWORD *)(v42 + 36) == 353 )
      {
        v15 = *(_QWORD *)&v11[-32 * v18];
        if ( v15 )
        {
          if ( a2 == v15 )
          {
LABEL_13:
            v16 = 0;
            goto LABEL_14;
          }
        }
      }
    }
    a3 = 32 * v18;
    if ( v6 < &v11[-a3] )
      goto LABEL_15;
    goto LABEL_81;
  }
  return 0;
}
