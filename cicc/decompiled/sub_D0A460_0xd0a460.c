// Function: sub_D0A460
// Address: 0xd0a460
//
__int64 __fastcall sub_D0A460(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  unsigned __int8 *v6; // rbx
  unsigned __int8 v7; // al
  unsigned __int8 v8; // dl
  int v9; // edx
  unsigned __int8 *v10; // r13
  __int64 v11; // rbx
  unsigned int v12; // r15d
  __int64 v14; // rax
  _BYTE *v15; // r13
  _QWORD *v16; // rax
  __int64 *v17; // rdx
  _BYTE *v18; // rcx
  __int64 *v19; // r15
  _BOOL4 v20; // r13d
  __int64 v21; // rax
  unsigned __int8 **v22; // r12
  unsigned __int8 **v23; // rbx
  __int64 v24; // rdx
  unsigned __int8 *v25; // r15
  _QWORD *v26; // rax
  __int64 *v27; // rdx
  _BOOL4 v28; // r15d
  __int64 v29; // rax
  _BYTE *v30; // rsi
  __int64 v31; // rdi
  unsigned __int8 *v32; // r12
  unsigned int v33; // r13d
  unsigned __int8 *v34; // rbx
  __int64 v35; // rdi
  unsigned __int8 *v36; // rax
  __int64 v37; // rax
  int v38; // edx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v45; // [rsp+30h] [rbp-F0h]
  int v46; // [rsp+30h] [rbp-F0h]
  __int64 *v47; // [rsp+38h] [rbp-E8h]
  __int64 v48; // [rsp+38h] [rbp-E8h]
  _BYTE *v49; // [rsp+40h] [rbp-E0h]
  unsigned __int8 v50; // [rsp+40h] [rbp-E0h]
  unsigned __int8 v51; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v54; // [rsp+60h] [rbp-C0h] BYREF
  unsigned __int8 *v55; // [rsp+68h] [rbp-B8h] BYREF
  __int64 v56; // [rsp+70h] [rbp-B0h] BYREF
  _BYTE *v57; // [rsp+78h] [rbp-A8h]
  _BYTE *v58; // [rsp+80h] [rbp-A0h]
  __int64 v59; // [rsp+90h] [rbp-90h] BYREF
  __int64 v60; // [rsp+98h] [rbp-88h] BYREF
  __int64 v61; // [rsp+A0h] [rbp-80h]
  __int64 *v62; // [rsp+A8h] [rbp-78h]
  __int64 *v63; // [rsp+B0h] [rbp-70h]
  __int64 v64; // [rsp+B8h] [rbp-68h]
  unsigned __int8 *v65; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v66; // [rsp+C8h] [rbp-58h]
  __int64 v67; // [rsp+D0h] [rbp-50h]
  __int64 v68; // [rsp+D8h] [rbp-48h]
  __int64 v69; // [rsp+E0h] [rbp-40h]
  __int64 v70; // [rsp+E8h] [rbp-38h]

  v5 = (__int64)a2;
  v6 = sub_98ACB0(*(unsigned __int8 **)a3, 6u);
  v7 = *v6;
  v8 = *v6;
  if ( (_BYTE)qword_4F86888 && !*(_BYTE *)(a4 + 360) && v7 == 84 )
  {
    v14 = sub_22077B0(8);
    v65 = v6;
    *(_QWORD *)v14 = v6;
    v15 = (_BYTE *)v14;
    v56 = v14;
    v58 = (_BYTE *)(v14 + 8);
    v57 = (_BYTE *)(v14 + 8);
    v49 = (_BYTE *)(v14 + 8);
    LODWORD(v60) = 0;
    v61 = 0;
    v62 = &v60;
    v63 = &v60;
    v64 = 0;
    v16 = sub_D0A360(&v59, &v60, (unsigned __int64 *)&v65);
    v18 = v49;
    v19 = v17;
    if ( v17 )
    {
      v20 = &v60 == v17 || v16 || v17[4] > (unsigned __int64)v6;
      v21 = sub_22077B0(40);
      *(_QWORD *)(v21 + 32) = v65;
      sub_220F040(v20, v21, v19, &v60);
      ++v64;
      v18 = v57;
      v15 = (_BYTE *)v56;
    }
    *(_BYTE *)(a4 + 360) = 1;
    v12 = 0;
    if ( v15 != v18 )
    {
      v50 = 0;
      do
      {
        v22 = (unsigned __int8 **)*((_QWORD *)v18 - 1);
        v18 -= 8;
        v57 = v18;
        if ( (*((_BYTE *)v22 + 7) & 0x40) != 0 )
        {
          v23 = (unsigned __int8 **)*(v22 - 1);
          v22 = &v23[4 * (*((_DWORD *)v22 + 1) & 0x7FFFFFF)];
        }
        else
        {
          v23 = &v22[-4 * (*((_DWORD *)v22 + 1) & 0x7FFFFFF)];
        }
        if ( v23 != v22 )
        {
          while ( 1 )
          {
            v54 = sub_98ACB0(*v23, 6u);
            v25 = v54;
            v26 = sub_D0A2C0((__int64)&v59, (unsigned __int64 *)&v54);
            if ( !v27 )
              goto LABEL_31;
            v28 = v26 || v27 == &v60 || (unsigned __int64)v25 < v27[4];
            v47 = v27;
            v29 = sub_22077B0(40);
            *(_QWORD *)(v29 + 32) = v54;
            sub_220F040(v28, v29, v47, &v60);
            ++v64;
            if ( *v54 == 84 )
            {
              v55 = v54;
              v30 = v57;
              if ( v57 == v58 )
              {
                sub_D09370((__int64)&v56, v57, &v55);
                goto LABEL_31;
              }
              if ( v57 )
              {
                *(_QWORD *)v57 = v54;
                v30 = v57;
              }
              v23 += 4;
              v57 = v30 + 8;
              if ( v22 == v23 )
              {
LABEL_41:
                v18 = v57;
                break;
              }
            }
            else
            {
              v65 = v54;
              v55 = 0;
              v24 = *(_QWORD *)(a3 + 8);
              v67 = 0;
              v68 = 0;
              v66 = v24;
              v69 = 0;
              v70 = 0;
              v50 |= sub_D0A460(a1, a2, &v65, a4);
              if ( v50 == 3 )
              {
                v12 = 3;
                goto LABEL_44;
              }
LABEL_31:
              v23 += 4;
              if ( v22 == v23 )
                goto LABEL_41;
            }
          }
        }
      }
      while ( (_BYTE *)v56 != v18 );
      v12 = v50;
    }
LABEL_44:
    sub_D00AB0(v61);
    if ( v56 )
      j_j___libc_free_0(v56, &v58[-v56]);
    return v12;
  }
  if ( v7 == 60 )
  {
    if ( *a2 != 85 )
      goto LABEL_97;
    if ( (*((_WORD *)a2 + 1) & 3u) - 1 <= 1 )
    {
      v65 = (unsigned __int8 *)*((_QWORD *)a2 + 9);
      if ( !(unsigned __int8)sub_A74390((__int64 *)&v65, 81, 0) )
        return 0;
      v7 = *v6;
    }
    v8 = v7;
    if ( v7 == 60 )
    {
LABEL_97:
      if ( !sub_B4D040((__int64)v6) && sub_D002E0((__int64)a2, 342) )
        return 2;
      v8 = *v6;
    }
  }
  if ( v8 <= 0x15u
    || v6 == a2
    || !(*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int8 *, unsigned __int8 *, _QWORD))(**(_QWORD **)(a4 + 344)
                                                                                                + 16LL))(
          *(_QWORD *)(a4 + 344),
          v6,
          a2,
          0)
    || *v6 != 60 && ((unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 53) || (unsigned __int8)sub_B49560((__int64)a2, 53)) )
  {
    goto LABEL_15;
  }
  v9 = *a2;
  v10 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v9 == 40 )
  {
    v11 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v11 = -32;
    if ( v9 != 85 )
    {
      v11 = -96;
      if ( v9 != 34 )
LABEL_92:
        BUG();
    }
  }
  if ( &a2[v11] == v10 )
    return 0;
  v51 = 0;
  v32 = v10;
  v33 = 0;
  v34 = &a2[v11];
  do
  {
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v32 + 8LL) + 8LL) != 14 )
      goto LABEL_57;
    if ( sub_CF49B0(a2, v33, 50) )
      goto LABEL_57;
    v35 = *(_QWORD *)a4;
    v66 = -1;
    v36 = *(unsigned __int8 **)a3;
    v67 = 0;
    v68 = 0;
    v65 = v36;
    v37 = *(_QWORD *)v32;
    v69 = 0;
    v70 = 0;
    v59 = v37;
    v60 = -1;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    if ( !(unsigned __int8)sub_CF4D50(v35, (__int64)&v59, (__int64)&v65, a4, 0) )
      goto LABEL_57;
    v38 = *a2;
    if ( v38 == 40 )
    {
      v48 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else
    {
      v48 = 0;
      if ( v38 != 85 )
      {
        if ( v38 != 34 )
          goto LABEL_92;
        v48 = 64;
      }
    }
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_72;
    v39 = sub_BD2BC0((__int64)a2);
    v45 = v40 + v39;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v45 >> 4) )
LABEL_94:
        BUG();
LABEL_72:
      v43 = 0;
      goto LABEL_73;
    }
    if ( !(unsigned int)((v45 - sub_BD2BC0((__int64)a2)) >> 4) )
      goto LABEL_72;
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_94;
    v46 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
    if ( (a2[7] & 0x80u) == 0 )
      BUG();
    v41 = sub_BD2BC0((__int64)a2);
    v43 = 32LL * (unsigned int)(*(_DWORD *)(v41 + v42 - 4) - v46);
LABEL_73:
    if ( v33 < (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v48 - v43) >> 5)
      && (unsigned __int8)sub_B49B80((__int64)a2, v33, 81)
      || sub_CF49B0(a2, v33, 51)
      || sub_CF49B0(a2, v33, 50) )
    {
      v51 |= 1u;
    }
    else
    {
      if ( !sub_CF49B0(a2, v33, 78) && !sub_CF49B0(a2, v33, 50) )
      {
        v5 = (__int64)a2;
        goto LABEL_15;
      }
      v51 |= 2u;
    }
LABEL_57:
    v32 += 32;
    ++v33;
  }
  while ( v32 != v34 );
  v5 = (__int64)a2;
  v12 = v51;
  if ( v51 != 3 )
    return v12;
LABEL_15:
  if ( (unsigned __int8)sub_D5CBF0(v5, *(_QWORD *)(a1 + 16)) )
  {
    v31 = *(_QWORD *)a4;
    v65 = (unsigned __int8 *)v5;
    v66 = -1;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    v70 = 0;
    if ( !(unsigned __int8)sub_CF4D50(v31, (__int64)&v65, a3, a4, 0) )
      return 0;
  }
  return !sub_D002E0(v5, 205) ? 3 : 1;
}
