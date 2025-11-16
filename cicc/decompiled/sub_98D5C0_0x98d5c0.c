// Function: sub_98D5C0
// Address: 0x98d5c0
//
__int64 __fastcall sub_98D5C0(unsigned __int8 *a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // eax
  __int64 v8; // rcx
  __int64 v9; // rax
  unsigned __int8 *v10; // rbx
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v15; // rdx
  int v16; // edx
  unsigned int v17; // r13d
  int v18; // eax
  __int64 v19; // r14
  __int64 v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // rdx
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r15
  int v36; // r15d
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rsi
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  int v43; // edx
  unsigned __int8 *v44; // rax
  __int64 v45; // rsi
  _QWORD *v46; // rax
  _QWORD *v47; // rdx

  v6 = *a1 - 29;
  v8 = v6;
  switch ( *a1 )
  {
    case 0x1Eu:
      v9 = sub_B43CB0(a1);
      if ( !(unsigned __int8)sub_B2D630(v9, 40) )
        goto LABEL_3;
      if ( (a1[7] & 0x40) != 0 )
        a3 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        a3 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v20 = *(_QWORD *)a3;
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_84;
      v44 = *(unsigned __int8 **)(a2 + 8);
      a3 = &v44[8 * *(unsigned int *)(a2 + 20)];
      if ( v44 == a3 )
        goto LABEL_3;
      while ( v20 != *(_QWORD *)v44 )
      {
        v44 += 8;
        if ( a3 == v44 )
          goto LABEL_3;
      }
      goto LABEL_13;
    case 0x1Fu:
      if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) != 3 )
        goto LABEL_15;
      v20 = *((_QWORD *)a1 - 12);
      if ( *(_BYTE *)(a2 + 28) )
      {
        v23 = *(_QWORD **)(a2 + 8);
        v24 = &v23[*(unsigned int *)(a2 + 20)];
        if ( v23 == v24 )
          goto LABEL_15;
        while ( v20 != *v23 )
        {
          if ( v24 == ++v23 )
            goto LABEL_15;
        }
      }
      else
      {
LABEL_84:
        if ( !sub_C8CA60(a2, v20, a3, v8, a5) )
          goto LABEL_3;
      }
      goto LABEL_13;
    case 0x20u:
      v20 = **((_QWORD **)a1 - 1);
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_84;
      v21 = *(_QWORD **)(a2 + 8);
      v22 = &v21[*(unsigned int *)(a2 + 20)];
      if ( v21 == v22 )
        goto LABEL_15;
      while ( v20 != *v21 )
      {
        if ( v22 == ++v21 )
          goto LABEL_15;
      }
      goto LABEL_13;
    case 0x22u:
    case 0x55u:
      if ( !(unsigned __int8)sub_B491E0() )
        goto LABEL_19;
      v45 = *((_QWORD *)a1 - 4);
      if ( *(_BYTE *)(a2 + 28) )
      {
        v46 = *(_QWORD **)(a2 + 8);
        v47 = &v46[*(unsigned int *)(a2 + 20)];
        if ( v46 != v47 )
        {
          while ( v45 != *v46 )
          {
            if ( v47 == ++v46 )
              goto LABEL_19;
          }
          goto LABEL_13;
        }
      }
      else if ( sub_C8CA60(a2, v45, v15, v8, a5) )
      {
        goto LABEL_13;
      }
LABEL_19:
      v16 = *a1;
      v17 = 0;
      v18 = v16 - 29;
      if ( v16 != 40 )
        goto LABEL_20;
LABEL_77:
      v19 = 32LL * (unsigned int)sub_B491D0(a1);
      if ( (a1[7] & 0x80u) == 0 )
        goto LABEL_78;
      while ( 1 )
      {
        v33 = sub_BD2BC0(a1);
        v35 = v33 + v34;
        if ( (a1[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v35 >> 4) )
LABEL_105:
            BUG();
LABEL_78:
          a3 = 0;
          goto LABEL_67;
        }
        if ( !(unsigned int)((v35 - sub_BD2BC0(a1)) >> 4) )
          goto LABEL_78;
        if ( (a1[7] & 0x80u) == 0 )
          goto LABEL_105;
        v36 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v37 = sub_BD2BC0(a1);
        a3 = (unsigned __int8 *)(32LL * (unsigned int)(*(_DWORD *)(v37 + v38 - 4) - v36));
LABEL_67:
        if ( v17 >= (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v19 - (__int64)a3) >> 5) )
          break;
        if ( (unsigned __int8)sub_B49B80(a1, v17, 40)
          || (unsigned __int8)sub_B49B80(a1, v17, 90)
          || (unsigned __int8)sub_B49B80(a1, v17, 91) )
        {
          v39 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
          v40 = *(_QWORD *)&a1[32 * (v17 - v39)];
          if ( *(_BYTE *)(a2 + 28) )
          {
            v41 = *(_QWORD **)(a2 + 8);
            v42 = &v41[*(unsigned int *)(a2 + 20)];
            if ( v41 != v42 )
            {
              while ( v40 != *v41 )
              {
                if ( v42 == ++v41 )
                  goto LABEL_76;
              }
              goto LABEL_13;
            }
          }
          else if ( sub_C8CA60(a2, v40, v39, v8, a5) )
          {
            goto LABEL_13;
          }
        }
LABEL_76:
        v43 = *a1;
        ++v17;
        v18 = v43 - 29;
        if ( v43 == 40 )
          goto LABEL_77;
LABEL_20:
        v19 = 0;
        if ( v18 != 56 )
        {
          if ( v18 != 5 )
            BUG();
          v19 = 64;
        }
        if ( (a1[7] & 0x80u) == 0 )
          goto LABEL_78;
      }
LABEL_3:
      v6 = *a1 - 29;
LABEL_4:
      if ( v6 <= 0x14 )
      {
        if ( v6 <= 0x12 )
        {
LABEL_15:
          LODWORD(a5) = 0;
          return (unsigned int)a5;
        }
LABEL_6:
        if ( (a1[7] & 0x40) != 0 )
          v10 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        else
          v10 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v11 = *((_QWORD *)v10 + 4);
        if ( !*(_BYTE *)(a2 + 28) )
        {
          LOBYTE(a5) = sub_C8CA60(a2, v11, a3, v8, a5) != 0;
          return (unsigned int)a5;
        }
        v12 = *(_QWORD **)(a2 + 8);
        v13 = &v12[*(unsigned int *)(a2 + 20)];
        if ( v12 == v13 )
          goto LABEL_15;
        while ( v11 != *v12 )
        {
          if ( v13 == ++v12 )
            goto LABEL_15;
        }
LABEL_13:
        LODWORD(a5) = 1;
        return (unsigned int)a5;
      }
      a5 = 0;
      if ( v6 - 22 <= 1 )
        goto LABEL_6;
      return (unsigned int)a5;
    case 0x3Du:
      v20 = *((_QWORD *)a1 - 4);
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_84;
      v25 = *(_QWORD **)(a2 + 8);
      v26 = &v25[*(unsigned int *)(a2 + 20)];
      if ( v25 == v26 )
        goto LABEL_15;
      while ( v20 != *v25 )
      {
        if ( v26 == ++v25 )
          goto LABEL_15;
      }
      goto LABEL_13;
    case 0x3Eu:
      v20 = *((_QWORD *)a1 - 4);
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_84;
      v27 = *(_QWORD **)(a2 + 8);
      v28 = &v27[*(unsigned int *)(a2 + 20)];
      if ( v27 == v28 )
        goto LABEL_15;
      while ( v20 != *v27 )
      {
        if ( v28 == ++v27 )
          goto LABEL_15;
      }
      goto LABEL_13;
    case 0x41u:
      v20 = *((_QWORD *)a1 - 12);
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_84;
      v29 = *(_QWORD **)(a2 + 8);
      v30 = &v29[*(unsigned int *)(a2 + 20)];
      if ( v29 == v30 )
        goto LABEL_15;
      while ( v20 != *v29 )
      {
        if ( v30 == ++v29 )
          goto LABEL_15;
      }
      goto LABEL_13;
    case 0x42u:
      v20 = *((_QWORD *)a1 - 8);
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_84;
      v31 = *(_QWORD **)(a2 + 8);
      v32 = &v31[*(unsigned int *)(a2 + 20)];
      if ( v31 == v32 )
        goto LABEL_15;
      while ( v20 != *v31 )
      {
        if ( v32 == ++v31 )
          goto LABEL_15;
      }
      goto LABEL_13;
    default:
      v8 = a4;
      goto LABEL_4;
  }
}
