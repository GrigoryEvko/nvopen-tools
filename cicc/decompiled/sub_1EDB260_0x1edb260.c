// Function: sub_1EDB260
// Address: 0x1edb260
//
__int64 __fastcall sub_1EDB260(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // r13
  __int64 v9; // r9
  __int64 v11; // rcx
  __int64 v12; // rax
  char v13; // r10
  int v14; // eax
  __int64 *v15; // r15
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r10
  unsigned int v19; // r11d
  unsigned int v20; // edi
  unsigned int *v21; // rsi
  char v22; // r15
  unsigned int *v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r10
  unsigned int *v26; // r10
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // rbx
  unsigned int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // r12
  int v35; // r14d
  unsigned int v36; // esi
  _QWORD *v37; // r13
  __int64 v38; // rdx
  unsigned int v39; // eax
  char v40; // al
  int v41; // r9d
  int v42; // eax
  bool v43; // al
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r10
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rax
  unsigned int *v52; // rax
  bool v53; // zf
  int v54; // eax
  _DWORD *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r10
  __int64 v59; // r9
  __int64 v60; // rsi
  __int64 v61; // rax
  int v62; // edx
  char v63; // [rsp+7h] [rbp-79h]
  __int64 v64; // [rsp+8h] [rbp-78h]
  int v65; // [rsp+8h] [rbp-78h]
  __int64 v67; // [rsp+10h] [rbp-70h]
  __int64 v68; // [rsp+10h] [rbp-70h]
  __int64 v69; // [rsp+10h] [rbp-70h]
  __int64 v70; // [rsp+18h] [rbp-68h]
  __int64 v71; // [rsp+18h] [rbp-68h]
  __int64 v72; // [rsp+18h] [rbp-68h]
  __int64 v73; // [rsp+20h] [rbp-60h]
  __int64 v74; // [rsp+20h] [rbp-60h]
  __int64 v75; // [rsp+20h] [rbp-60h]
  __int64 v76; // [rsp+20h] [rbp-60h]
  __int64 v77; // [rsp+20h] [rbp-60h]
  __int64 v78; // [rsp+20h] [rbp-60h]
  __int64 v79; // [rsp+20h] [rbp-60h]
  unsigned __int64 v80; // [rsp+28h] [rbp-58h]
  __int64 v81; // [rsp+28h] [rbp-58h]
  unsigned int *v82; // [rsp+28h] [rbp-58h]
  __int64 v83; // [rsp+28h] [rbp-58h]
  __int64 v84; // [rsp+28h] [rbp-58h]
  __int64 v85; // [rsp+28h] [rbp-58h]
  __int64 v86; // [rsp+28h] [rbp-58h]
  unsigned int *v87; // [rsp+30h] [rbp-50h] BYREF

  v5 = a3;
  v6 = a2;
  result = *(_QWORD *)(a1 + 112);
  v8 = result + 40LL * a2;
  LODWORD(v9) = *(_DWORD *)(v8 + 4);
  if ( (_DWORD)v9 )
    return result;
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 64LL) + 8LL * a2);
  v12 = *(_QWORD *)(v11 + 8);
  if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    *(_QWORD *)v8 = 0xFFFFFFFF00000000LL;
    goto LABEL_26;
  }
  v13 = *(_BYTE *)(a1 + 20);
  if ( (v12 & 6) != 0 )
  {
    v27 = *(_QWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v70 = v27;
    if ( v13 )
    {
      *(_DWORD *)(v8 + 8) = 1;
      *(_DWORD *)(v8 + 4) = 1;
      if ( **(_WORD **)(v27 + 16) == 9 )
      {
        *(_DWORD *)(v8 + 8) = 0;
        *(_BYTE *)(v8 + 32) = 1;
      }
    }
    else
    {
      v33 = *(_QWORD *)(v27 + 32);
      if ( v33 == v33 + 40LL * *(unsigned int *)(v27 + 40) )
      {
        *(_DWORD *)(v8 + 8) = 0;
      }
      else
      {
        v75 = v8;
        v83 = a2;
        v34 = v33 + 40LL * *(unsigned int *)(v27 + 40);
        v35 = 0;
        do
        {
          if ( !*(_BYTE *)v33 && *(_DWORD *)(v33 + 8) == *(_DWORD *)(a1 + 8) && (*(_BYTE *)(v33 + 3) & 0x10) != 0 )
          {
            v36 = *(_DWORD *)(a1 + 12);
            v37 = *(_QWORD **)(a1 + 56);
            v38 = (*(_DWORD *)v33 >> 8) & 0xFFF;
            if ( v36 )
            {
              if ( (_DWORD)v38 )
              {
                v63 = v13;
                v64 = v11;
                v39 = (*(__int64 (__fastcall **)(_QWORD *))(*v37 + 120LL))(v37);
                v11 = v64;
                v13 = v63;
                v38 = v39;
              }
              else
              {
                v38 = v36;
              }
            }
            v35 |= *(_DWORD *)(v37[31] + 4 * v38);
            v40 = *(_BYTE *)(v33 + 4);
            if ( (v40 & 1) == 0 && (v40 & 2) == 0 )
            {
              if ( (*(_BYTE *)(v33 + 3) & 0x10) != 0 )
              {
                if ( (*(_DWORD *)v33 & 0xFFF00) != 0 )
                  v13 = 1;
              }
              else
              {
                v13 = 1;
              }
            }
          }
          v33 += 40;
        }
        while ( v34 != v33 );
        v8 = v75;
        v41 = v35;
        v6 = v83;
        v5 = a3;
        *(_DWORD *)(v75 + 4) = v41;
        *(_DWORD *)(v75 + 8) = v41;
        if ( v13 )
        {
          v86 = v11;
          sub_1E86030((__int64)&v87, *(_QWORD *)a1, *(_QWORD *)(v11 + 8));
          v52 = v87;
          v11 = v86;
          v53 = v87 == 0;
          *(_QWORD *)(v75 + 16) = v87;
          if ( !v53 )
          {
            sub_1EDB260(a1, *v52, a3, v86);
            v11 = v86;
            *(_DWORD *)(v75 + 8) |= *(_DWORD *)(*(_QWORD *)(a1 + 112) + 40LL * **(unsigned int **)(v75 + 16) + 8);
          }
        }
      }
      if ( **(_WORD **)(v70 + 16) == 9 )
      {
        v42 = *(_DWORD *)(v8 + 4);
        *(_BYTE *)(v8 + 32) = 1;
        *(_DWORD *)(v8 + 8) &= ~v42;
      }
    }
  }
  else
  {
    v14 = 1;
    if ( !v13 )
      v14 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 248LL) + 4LL * *(unsigned int *)(a1 + 12));
    *(_DWORD *)(v8 + 4) = v14;
    *(_DWORD *)(v8 + 8) = v14;
    v70 = 0;
  }
  v15 = *(__int64 **)v5;
  v73 = v11;
  v80 = *(_QWORD *)(v11 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v16 = (__int64 *)sub_1DB3C70(*(__int64 **)v5, v80);
  v17 = v73;
  v18 = *v15 + 24LL * *((unsigned int *)v15 + 2);
  if ( v16 == (__int64 *)v18 )
  {
    *(_QWORD *)(v8 + 24) = 0;
  }
  else
  {
    v19 = *(_DWORD *)(v80 + 24);
    v20 = *(_DWORD *)((*v16 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( (unsigned __int64)(v20 | (*v16 >> 1) & 3) > v19 )
    {
      v22 = 0;
      v9 = 0;
      v21 = 0;
      if ( v19 < v20 )
        goto LABEL_13;
    }
    else
    {
      v9 = v16[1];
      v21 = (unsigned int *)v16[2];
      v22 = 0;
      a5 = v9 & 0xFFFFFFF8;
      if ( v80 == (v9 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        if ( (__int64 *)v18 == v16 + 3 )
        {
          v22 = 1;
          goto LABEL_13;
        }
        v20 = *(_DWORD *)((v16[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v16 += 3;
        v22 = 1;
      }
      if ( v80 == *((_QWORD *)v21 + 1) )
        v21 = 0;
      if ( v19 < v20 )
      {
LABEL_13:
        v74 = v9;
        v81 = v17;
        *(_QWORD *)(v8 + 24) = v21;
        if ( v21 )
        {
          sub_1EDB260(v5, *v21, a1, v17);
          v23 = *(unsigned int **)(v8 + 24);
          v24 = v81;
          v9 = v74;
          v25 = *(_QWORD *)(v5 + 112) + 40LL * *v23;
          if ( *(_BYTE *)(v25 + 32) )
          {
            if ( v70 )
            {
              v68 = v74;
              v77 = *(_QWORD *)(v5 + 112) + 40LL * *v23;
              v51 = sub_1DA9310(*(_QWORD *)(a1 + 48), *((_QWORD *)v23 + 1));
              v25 = v77;
              v9 = v68;
              v24 = v81;
              if ( *(_QWORD *)(v70 + 24) != v51 )
                *(_BYTE *)(v77 + 32) = 0;
            }
          }
          if ( (*(_BYTE *)(v24 + 8) & 6) == 0 )
            goto LABEL_78;
          if ( **(_WORD **)(v70 + 16) == 9 )
          {
            if ( !*(_BYTE *)(a1 + 21) || (*(_DWORD *)(v8 + 4) & (*(_DWORD *)(v25 + 8) | *(_DWORD *)(v25 + 4))) != 0 )
            {
              *(_DWORD *)v8 = 1;
              v26 = *(unsigned int **)(v8 + 24);
LABEL_40:
              result = *(_QWORD *)(a1 + 64);
              *(_DWORD *)(result + 4 * v6) = *(_DWORD *)(*(_QWORD *)(v5 + 64) + 4LL * *v26);
              return result;
            }
            goto LABEL_78;
          }
          v67 = v9;
          v76 = v25;
          v84 = v24;
          v43 = sub_1EDB0A0(*(unsigned int **)(a1 + 32), v70);
          v45 = v84;
          v46 = v76;
          v9 = v67;
          if ( v43 )
          {
            v54 = *(_DWORD *)(v76 + 8) | ~*(_DWORD *)(v8 + 4);
            *(_DWORD *)v8 = 1;
            *(_DWORD *)(v8 + 8) &= v54;
            v26 = *(unsigned int **)(v8 + 24);
            goto LABEL_40;
          }
          if ( !v22
            || (v44 = *(_DWORD *)((v67 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v67 >> 1) & 3,
                (unsigned int)v44 > (*(_DWORD *)((*(_QWORD *)(v84 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                   | (unsigned int)(*(__int64 *)(v84 + 8) >> 1) & 3)) )
          {
            if ( **(_WORD **)(v70 + 16) == 15 )
            {
              v55 = *(_DWORD **)(v70 + 32);
              if ( (*v55 & 0xFFF00) == 0 && (v55[10] & 0xFFF00) == 0 && !*(_BYTE *)(*(_QWORD *)(a1 + 32) + 24LL) )
              {
                v71 = v76;
                v78 = *(_QWORD *)(v8 + 24);
                v56 = sub_1ED97C0(a1, v84, v44, v84, a5);
                v58 = v71;
                v59 = v67;
                v65 = v57;
                if ( v78 == v56 && (_DWORD)v57 == *(_DWORD *)(v5 + 8) )
                  goto LABEL_94;
                v60 = v78;
                v69 = v56;
                v72 = v59;
                v79 = v58;
                v61 = sub_1ED97C0(v5, v60, v57, v84, v56);
                a5 = v69;
                v45 = v84;
                v46 = v79;
                v9 = v72;
                if ( v69 )
                {
                  if ( v61 && *(_QWORD *)(v61 + 8) == *(_QWORD *)(v69 + 8) && v65 == v62 )
                  {
LABEL_94:
                    *(_BYTE *)(v8 + 35) = 1;
                    v26 = *(unsigned int **)(v8 + 24);
                    *(_DWORD *)v8 = 1;
                    goto LABEL_40;
                  }
                }
                else if ( v65 == v62 && !v61 )
                {
                  goto LABEL_94;
                }
              }
            }
            if ( (*(_DWORD *)(v46 + 8) & *(_DWORD *)(v8 + 4)) != 0 )
            {
              v85 = v9;
              if ( !v22
                && (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 248LL) + 4LL * *(unsigned int *)(v5 + 12))
                  & ~*(_DWORD *)(v8 + 4)) != 0 )
              {
                v47 = *(_QWORD *)(a1 + 48);
                v48 = sub_1DA9310(v47, *(_QWORD *)(v45 + 8));
                LODWORD(v9) = v85;
                v49 = *(_QWORD *)(*(_QWORD *)(v47 + 392) + 16LL * *(unsigned int *)(v48 + 48) + 8);
                if ( (*(_DWORD *)((v85 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v85 >> 1) & 3) < (*(_DWORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v49 >> 1) & 3) )
                {
                  *(_DWORD *)v8 = 4;
                  goto LABEL_69;
                }
              }
              goto LABEL_41;
            }
LABEL_78:
            *(_DWORD *)v8 = 3;
LABEL_69:
            v50 = *(_QWORD *)(v5 + 112) + 40LL * **(unsigned int **)(v8 + 24);
            if ( (*(_DWORD *)(v50 + 4) & ~*(_DWORD *)(v8 + 8)) != 0 && *(_BYTE *)(a1 + 21) )
              *(_BYTE *)(v50 + 32) = 0;
            *(_BYTE *)(v50 + 33) = 1;
            goto LABEL_26;
          }
        }
        goto LABEL_25;
      }
    }
    v26 = (unsigned int *)v16[2];
    v9 = v16[1];
    if ( !v26 || v26 == v21 )
      goto LABEL_13;
    v30 = *(_DWORD *)((*((_QWORD *)v26 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*((__int64 *)v26 + 1) >> 1) & 3;
    v31 = *(_DWORD *)((*(_QWORD *)(v73 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v73 + 8) >> 1) & 3;
    if ( v30 < v31 )
    {
      v82 = v26;
      sub_1EDB260(v5, *v26, a1, v73);
      v26 = v82;
      v17 = v73;
    }
    else if ( v30 > v31 && v21 )
    {
      *(_QWORD *)(v8 + 24) = v21;
      *(_DWORD *)v8 = 5;
      goto LABEL_26;
    }
    *(_QWORD *)(v8 + 24) = v26;
    v32 = *(_QWORD *)(v5 + 112) + 40LL * *v26;
    if ( *(_DWORD *)(v32 + 4) )
    {
      if ( (*(_BYTE *)(v17 + 8) & 6) == 0 || (*(_DWORD *)(v8 + 8) & *(_DWORD *)(v32 + 8)) == 0 )
      {
        *(_DWORD *)v8 = 2;
        goto LABEL_40;
      }
LABEL_41:
      *(_DWORD *)v8 = 5;
      goto LABEL_26;
    }
  }
LABEL_25:
  *(_DWORD *)v8 = 0;
LABEL_26:
  *(_DWORD *)(*(_QWORD *)(a1 + 64) + 4 * v6) = *(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
  v28 = *(_QWORD *)(a1 + 24);
  v29 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 64LL) + 8 * v6);
  result = *(unsigned int *)(v28 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(v28 + 12) )
  {
    sub_16CD150(v28, (const void *)(v28 + 16), 0, 8, a5, v9);
    result = *(unsigned int *)(v28 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v28 + 8 * result) = v29;
  ++*(_DWORD *)(v28 + 8);
  return result;
}
