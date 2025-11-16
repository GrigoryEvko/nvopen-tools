// Function: sub_BD22F0
// Address: 0xbd22f0
//
__int64 __fastcall sub_BD22F0(__int64 a1, _QWORD *a2, char a3)
{
  _QWORD *v4; // r12
  _QWORD *v5; // rbx
  _QWORD *i; // rbx
  _QWORD *j; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // rbx
  _QWORD *v11; // r12
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  _QWORD *v14; // r13
  unsigned __int8 **v15; // rbx
  unsigned __int8 **v16; // r14
  unsigned __int8 *v17; // rsi
  int v18; // eax
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  _BYTE *v21; // r12
  _BYTE *v22; // rbx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rdx
  _QWORD *v28; // rbx
  _QWORD *v29; // r13
  __int64 result; // rax
  __int64 v31; // rbx
  _QWORD *k; // r12
  unsigned int v33; // r13d
  int v34; // r14d
  unsigned int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // rcx
  _QWORD *v38; // [rsp+20h] [rbp-D0h]
  _QWORD *v39; // [rsp+28h] [rbp-C8h]
  _QWORD *v40; // [rsp+30h] [rbp-C0h]
  _QWORD *v41; // [rsp+50h] [rbp-A0h]
  __int64 v42; // [rsp+58h] [rbp-98h]
  __int64 v43; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v44; // [rsp+68h] [rbp-88h]
  _BYTE *v45; // [rsp+70h] [rbp-80h] BYREF
  __int64 v46; // [rsp+78h] [rbp-78h]
  _BYTE v47[112]; // [rsp+80h] [rbp-70h] BYREF

  v4 = a2 + 1;
  *(_BYTE *)(a1 + 152) = a3;
  v5 = (_QWORD *)a2[2];
  v38 = a2;
  if ( v5 != a2 + 1 )
  {
    do
    {
      if ( !v5 )
        BUG();
      a2 = (_QWORD *)*(v5 - 4);
      sub_BD0F10(a1, (__int64)a2);
      if ( !sub_B2FC80((__int64)(v5 - 7)) )
      {
        a2 = (_QWORD *)*(v5 - 11);
        sub_BD1B10(a1, (unsigned __int8 *)a2);
      }
      v5 = (_QWORD *)v5[1];
    }
    while ( v4 != v5 );
  }
  for ( i = (_QWORD *)v38[6]; i != v38 + 5; i = (_QWORD *)i[1] )
  {
    if ( !i )
      BUG();
    sub_BD0F10(a1, *(i - 3));
    a2 = (_QWORD *)*(i - 10);
    if ( a2 )
      sub_BD1B10(a1, (unsigned __int8 *)a2);
  }
  for ( j = (_QWORD *)v38[8]; j != v38 + 7; j = (_QWORD *)j[1] )
  {
    if ( !j )
      BUG();
    a2 = (_QWORD *)*(j - 4);
    sub_BD0F10(a1, (__int64)a2);
  }
  v45 = v47;
  v46 = 0x400000000LL;
  v40 = (_QWORD *)v38[4];
  if ( v40 == v38 + 3 )
  {
    result = (__int64)v38;
    v31 = v38[10];
    k = v38 + 9;
    if ( v38 + 9 == (_QWORD *)v31 )
      return result;
    goto LABEL_63;
  }
  do
  {
    if ( !v40 )
      BUG();
    sub_BD0F10(a1, *(v40 - 4));
    a2 = (_QWORD *)v40[8];
    sub_BD2020(a1, (__int64)a2);
    if ( (*((_BYTE *)v40 - 49) & 0x40) != 0 )
    {
      v10 = (_QWORD *)*(v40 - 8);
      v11 = &v10[4 * (*((_DWORD *)v40 - 13) & 0x7FFFFFF)];
    }
    else
    {
      v11 = v40 - 7;
      v10 = &v40[-4 * (*((_DWORD *)v40 - 13) & 0x7FFFFFF) - 7];
    }
    while ( v11 != v10 )
    {
      a2 = (_QWORD *)*v10;
      v10 += 4;
      sub_BD1B10(a1, (unsigned __int8 *)a2);
    }
    if ( (*((_BYTE *)v40 - 54) & 1) != 0 )
    {
      sub_B2C6D0((__int64)(v40 - 7), (__int64)a2, v8, v9);
      v12 = (_QWORD *)v40[5];
      v13 = &v12[5 * v40[6]];
      if ( (*((_BYTE *)v40 - 54) & 1) != 0 )
      {
        sub_B2C6D0((__int64)(v40 - 7), (__int64)a2, v36, v37);
        v12 = (_QWORD *)v40[5];
      }
    }
    else
    {
      v12 = (_QWORD *)v40[5];
      v13 = &v12[5 * v40[6]];
    }
    while ( v13 != v12 )
    {
      a2 = v12;
      v12 += 5;
      sub_BD1B10(a1, (unsigned __int8 *)a2);
    }
    v39 = (_QWORD *)v40[3];
    if ( v40 + 2 != v39 )
    {
LABEL_25:
      if ( !v39 )
        BUG();
      v14 = (_QWORD *)v39[4];
      if ( v39 + 3 == v14 )
        goto LABEL_60;
      while ( 1 )
      {
        if ( !v14 )
          BUG();
        sub_BD0F10(a1, *(v14 - 2));
        if ( (*((_BYTE *)v14 - 17) & 0x40) != 0 )
        {
          v15 = (unsigned __int8 **)*(v14 - 4);
          v16 = &v15[4 * (*((_DWORD *)v14 - 5) & 0x7FFFFFF)];
        }
        else
        {
          v16 = (unsigned __int8 **)(v14 - 3);
          v15 = (unsigned __int8 **)&v14[-4 * (*((_DWORD *)v14 - 5) & 0x7FFFFFF) - 3];
        }
        while ( v16 != v15 )
        {
          while ( 1 )
          {
            v17 = *v15;
            if ( *v15 )
            {
              if ( *v17 <= 0x1Cu )
                break;
            }
            v15 += 4;
            if ( v16 == v15 )
              goto LABEL_36;
          }
          v15 += 4;
          sub_BD1B10(a1, v17);
        }
LABEL_36:
        v18 = *((unsigned __int8 *)v14 - 24);
        if ( (_BYTE)v18 == 63 )
        {
          sub_BD0F10(a1, v14[6]);
          v18 = *((unsigned __int8 *)v14 - 24);
        }
        if ( (_BYTE)v18 == 60 )
        {
          sub_BD0F10(a1, v14[6]);
          v18 = *((unsigned __int8 *)v14 - 24);
        }
        v19 = (unsigned int)(v18 - 34);
        if ( (unsigned __int8)v19 <= 0x33u )
        {
          v20 = 0x8000000000041LL;
          if ( _bittest64(&v20, v19) )
            sub_BD2020(a1, v14[6]);
        }
        a2 = &v45;
        sub_B9A9D0((__int64)(v14 - 3), (__int64)&v45);
        v21 = v45;
        v22 = &v45[16 * (unsigned int)v46];
        if ( v22 != v45 )
        {
          do
          {
            a2 = (_QWORD *)*((_QWORD *)v21 + 1);
            v21 += 16;
            sub_BD1850(a1, (__int64)a2);
          }
          while ( v21 != v22 );
        }
        v23 = v14[5];
        LODWORD(v46) = 0;
        if ( v23 )
        {
          v24 = sub_B14240(v23);
          v42 = v25;
          v26 = v24;
          if ( v24 != v25 )
            break;
        }
LABEL_59:
        v14 = (_QWORD *)v14[1];
        if ( v39 + 3 == v14 )
        {
LABEL_60:
          v39 = (_QWORD *)v39[1];
          if ( v40 + 2 == v39 )
            goto LABEL_61;
          goto LABEL_25;
        }
      }
      v41 = v14;
      while ( 1 )
      {
        while ( *(_BYTE *)(v26 + 32) )
        {
LABEL_57:
          v26 = *(_QWORD *)(v26 + 8);
          if ( v26 == v42 )
          {
LABEL_58:
            v14 = v41;
            goto LABEL_59;
          }
        }
        a2 = (_QWORD *)v26;
        sub_B129C0(&v43, v26);
        v27 = v43;
        v28 = v44;
        if ( v44 != (_QWORD *)v43 )
          break;
LABEL_56:
        if ( *(_BYTE *)(v26 + 64) != 2 )
          goto LABEL_57;
        a2 = sub_B13320(v26);
        if ( !a2 )
          goto LABEL_57;
        sub_BD1B10(a1, (unsigned __int8 *)a2);
        v26 = *(_QWORD *)(v26 + 8);
        if ( v26 == v42 )
          goto LABEL_58;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v29 = (_QWORD *)(v27 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v27 & 4) == 0 )
            break;
          a2 = *(_QWORD **)(*v29 + 136LL);
          sub_BD1B10(a1, (unsigned __int8 *)a2);
LABEL_52:
          v27 = (unsigned __int64)(v29 + 1) | 4;
          if ( (_QWORD *)v27 == v28 )
            goto LABEL_56;
        }
        a2 = (_QWORD *)v29[17];
        sub_BD1B10(a1, (unsigned __int8 *)a2);
        v27 = (__int64)(v29 + 18);
        if ( !v29 )
          goto LABEL_52;
        if ( v29 + 18 == v28 )
          goto LABEL_56;
      }
    }
LABEL_61:
    v40 = (_QWORD *)v40[1];
  }
  while ( v38 + 3 != v40 );
  result = (__int64)v38;
  v31 = v38[10];
  for ( k = v38 + 9; k != (_QWORD *)v31; v31 = *(_QWORD *)(v31 + 8) )
  {
LABEL_63:
    v33 = 0;
    result = sub_B91A00(v31);
    v34 = result;
    if ( (_DWORD)result )
    {
      do
      {
        v35 = v33++;
        a2 = (_QWORD *)sub_B91A10(v31, v35);
        result = sub_BD1850(a1, (__int64)a2);
      }
      while ( v34 != v33 );
    }
  }
  if ( v45 != v47 )
    return _libc_free(v45, a2);
  return result;
}
