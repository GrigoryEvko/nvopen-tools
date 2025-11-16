// Function: sub_18910C0
// Address: 0x18910c0
//
__int64 __fastcall sub_18910C0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // r13
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 *v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rbx
  bool v21; // cc
  __int64 v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // r14
  __int64 v27; // rdi
  unsigned __int64 *v28; // r13
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rsi
  __int64 v44; // rdx
  unsigned __int8 *v45; // rsi
  int v46; // edi
  __int64 v47; // rax
  __int64 v48; // [rsp+10h] [rbp-A0h]
  unsigned __int64 *v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  unsigned int v51; // [rsp+24h] [rbp-8Ch]
  __int64 v52; // [rsp+28h] [rbp-88h]
  __int64 v53; // [rsp+28h] [rbp-88h]
  __int64 v54; // [rsp+28h] [rbp-88h]
  unsigned int v55; // [rsp+34h] [rbp-7Ch] BYREF
  unsigned __int8 *v56; // [rsp+38h] [rbp-78h] BYREF
  __int64 v57[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v58; // [rsp+50h] [rbp-60h]
  _BYTE v59[16]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v60; // [rsp+70h] [rbp-40h]

  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v7 != 13 )
  {
    if ( v7 == 11 )
    {
      if ( *(_BYTE *)(a3 + 8) != 15 )
        goto LABEL_5;
      v58 = 257;
      if ( a3 == v6 )
        return a2;
      if ( *(_BYTE *)(a2 + 16) <= 0x10u )
        return sub_15A46C0(46, (__int64 ***)a2, (__int64 **)a3, 0);
      v60 = 257;
      v46 = 46;
    }
    else
    {
      if ( v7 != 15 || *(_BYTE *)(a3 + 8) != 11 )
      {
LABEL_5:
        v58 = 257;
        if ( a3 != v6 )
        {
          if ( *(_BYTE *)(a2 + 16) <= 0x10u )
            return sub_15A46C0(47, (__int64 ***)a2, (__int64 **)a3, 0);
          v60 = 257;
          v10 = sub_15FDBD0(47, a2, a3, (__int64)v59, 0);
          v11 = a1[1];
          v8 = v10;
          if ( !v11 )
            goto LABEL_16;
          v12 = (__int64 *)a1[2];
          v13 = v11 + 40;
          goto LABEL_15;
        }
        return a2;
      }
      v58 = 257;
      if ( a3 == v6 )
        return a2;
      if ( *(_BYTE *)(a2 + 16) <= 0x10u )
        return sub_15A46C0(45, (__int64 ***)a2, (__int64 **)a3, 0);
      v60 = 257;
      v46 = 45;
    }
    v8 = sub_15FDBD0(v46, a2, a3, (__int64)v59, 0);
    v47 = a1[1];
    if ( !v47 )
    {
LABEL_16:
      sub_164B780(v8, v57);
      v16 = *a1;
      if ( *a1 )
      {
        v56 = (unsigned __int8 *)*a1;
        sub_1623A60((__int64)&v56, v16, 2);
        v17 = *(_QWORD *)(v8 + 48);
        if ( v17 )
          sub_161E7C0(v8 + 48, v17);
        v18 = v56;
        *(_QWORD *)(v8 + 48) = v56;
        if ( v18 )
          sub_1623210((__int64)&v56, v18, v8 + 48);
      }
      return v8;
    }
    v12 = (__int64 *)a1[2];
    v13 = v47 + 40;
LABEL_15:
    sub_157E9D0(v13, v8);
    v14 = *v12;
    v15 = *(_QWORD *)(v8 + 24);
    *(_QWORD *)(v8 + 32) = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v14 | v15 & 7;
    *(_QWORD *)(v14 + 8) = v8 + 24;
    *v12 = *v12 & 7 | (v8 + 24);
    goto LABEL_16;
  }
  v55 = 0;
  v8 = sub_1599EF0((__int64 **)a3);
  v51 = *(_DWORD *)(v6 + 12);
  if ( v51 )
  {
    v19 = 0;
    v20 = a3;
    do
    {
      v21 = *(_BYTE *)(a2 + 16) <= 0x10u;
      v22 = *(_QWORD *)(*(_QWORD *)(v20 + 16) + 8 * v19);
      v58 = 257;
      v52 = v22;
      if ( v21 )
      {
        v23 = (_QWORD *)sub_15A3AE0((_QWORD *)a2, &v55, 1, 0);
      }
      else
      {
        v60 = 257;
        v34 = sub_1648A60(88, 1u);
        v23 = v34;
        if ( v34 )
        {
          v50 = (__int64)v34;
          v35 = sub_15FB2A0(*(_QWORD *)a2, &v55, 1);
          sub_15F1EA0((__int64)v23, v35, 62, (__int64)(v23 - 3), 1, 0);
          if ( *(v23 - 3) )
          {
            v36 = *(v23 - 2);
            v37 = *(v23 - 1) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v37 = v36;
            if ( v36 )
              *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
          }
          *(v23 - 3) = a2;
          v38 = *(_QWORD *)(a2 + 8);
          *(v23 - 2) = v38;
          if ( v38 )
            *(_QWORD *)(v38 + 16) = (unsigned __int64)(v23 - 2) | *(_QWORD *)(v38 + 16) & 3LL;
          *(v23 - 1) = (a2 + 8) | *(v23 - 1) & 3LL;
          *(_QWORD *)(a2 + 8) = v23 - 3;
          v23[7] = v23 + 9;
          v23[8] = 0x400000000LL;
          sub_15FB110((__int64)v23, &v55, 1, (__int64)v59);
        }
        else
        {
          v50 = 0;
        }
        v39 = a1[1];
        if ( v39 )
        {
          v49 = (unsigned __int64 *)a1[2];
          sub_157E9D0(v39 + 40, (__int64)v23);
          v40 = *v49;
          v41 = v23[3] & 7LL;
          v23[4] = v49;
          v40 &= 0xFFFFFFFFFFFFFFF8LL;
          v23[3] = v40 | v41;
          *(_QWORD *)(v40 + 8) = v23 + 3;
          *v49 = *v49 & 7 | (unsigned __int64)(v23 + 3);
        }
        sub_164B780(v50, v57);
        v42 = *a1;
        if ( *a1 )
        {
          v56 = (unsigned __int8 *)*a1;
          sub_1623A60((__int64)&v56, v42, 2);
          v43 = v23[6];
          v44 = (__int64)(v23 + 6);
          if ( v43 )
          {
            sub_161E7C0((__int64)(v23 + 6), v43);
            v44 = (__int64)(v23 + 6);
          }
          v45 = v56;
          v23[6] = v56;
          if ( v45 )
            sub_1623210((__int64)&v56, v45, v44);
        }
      }
      v24 = sub_18910C0(a1, v23, v52);
      v58 = 257;
      if ( *(_BYTE *)(v8 + 16) > 0x10u || *(_BYTE *)(v24 + 16) > 0x10u )
      {
        v53 = v24;
        v60 = 257;
        v25 = sub_1648A60(88, 2u);
        v26 = v25;
        if ( v25 )
        {
          v48 = v53;
          v54 = (__int64)v25;
          sub_15F1EA0((__int64)v25, *(_QWORD *)v8, 63, (__int64)(v25 - 6), 2, 0);
          v26[7] = v26 + 9;
          v26[8] = 0x400000000LL;
          sub_15FAD90((__int64)v26, v8, v48, &v55, 1, (__int64)v59);
        }
        else
        {
          v54 = 0;
        }
        v27 = a1[1];
        if ( v27 )
        {
          v28 = (unsigned __int64 *)a1[2];
          sub_157E9D0(v27 + 40, (__int64)v26);
          v29 = v26[3];
          v30 = *v28;
          v26[4] = v28;
          v30 &= 0xFFFFFFFFFFFFFFF8LL;
          v26[3] = v30 | v29 & 7;
          *(_QWORD *)(v30 + 8) = v26 + 3;
          *v28 = *v28 & 7 | (unsigned __int64)(v26 + 3);
        }
        sub_164B780(v54, v57);
        v31 = *a1;
        if ( *a1 )
        {
          v56 = (unsigned __int8 *)*a1;
          sub_1623A60((__int64)&v56, v31, 2);
          v32 = v26[6];
          if ( v32 )
            sub_161E7C0((__int64)(v26 + 6), v32);
          v33 = v56;
          v26[6] = v56;
          if ( v33 )
            sub_1623210((__int64)&v56, v33, (__int64)(v26 + 6));
        }
        v8 = (__int64)v26;
      }
      else
      {
        v8 = sub_15A3A20((__int64 *)v8, (__int64 *)v24, &v55, 1, 0);
      }
      v19 = v55 + 1;
      v55 = v19;
    }
    while ( (unsigned int)v19 < v51 );
  }
  return v8;
}
