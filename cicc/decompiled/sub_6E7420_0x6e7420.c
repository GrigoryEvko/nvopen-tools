// Function: sub_6E7420
// Address: 0x6e7420
//
__int64 __fastcall sub_6E7420(
        __int64 a1,
        _DWORD *a2,
        _BOOL4 a3,
        int a4,
        int a5,
        int a6,
        char a7,
        __int64 *a8,
        _DWORD *a9,
        _DWORD *a10)
{
  _DWORD *v10; // r14
  int v13; // eax
  __int64 v14; // rdi
  bool v15; // zf
  _BOOL4 v16; // eax
  __int64 result; // rax
  __int64 v18; // r15
  _QWORD *v19; // rbx
  _QWORD *v20; // rax
  __int64 v21; // r13
  char v22; // r12
  char v23; // al
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 i; // rdi
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 *v30; // r13
  char v31; // dl
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-60h]
  char v35; // [rsp+13h] [rbp-4Dh]
  unsigned int v36; // [rsp+14h] [rbp-4Ch]
  _QWORD *v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+28h] [rbp-38h]
  __int64 v41; // [rsp+28h] [rbp-38h]

  v10 = a2;
  if ( a10 )
    *a10 = 0;
  v13 = sub_6E6010();
  v14 = dword_4F077BC;
  if ( !dword_4F077BC || !qword_4D03C50 || (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) == 0 )
  {
    v15 = v13 == 0;
    v16 = 0;
    if ( !v15 )
      v16 = a3;
    a3 = v16;
  }
  if ( (*(_BYTE *)(a1 + 96) & 4) == 0 )
  {
LABEL_16:
    v18 = *(_QWORD *)*a8;
    v33 = v18;
    v36 = sub_8D2EF0(v18);
    if ( v36 )
    {
      v36 = 1;
      v18 = sub_8D46C0(v18);
    }
    while ( *(_BYTE *)(v18 + 140) == 12 )
      v18 = *(_QWORD *)(v18 + 160);
    v39 = 0;
    v19 = *(_QWORD **)(a1 + 112);
    if ( v19 )
    {
      v20 = (_QWORD *)v19[2];
      v39 = v20;
      if ( (*(_BYTE *)(a1 + 96) & 2) == 0 )
        v20 = (_QWORD *)v19[1];
      v19 = v20;
    }
    v35 = (a7 & 1) << 6;
    if ( !v19 )
      goto LABEL_45;
LABEL_25:
    v21 = v19[2];
    v22 = 1;
    if ( !a3 )
      goto LABEL_33;
    v23 = *(_BYTE *)(v21 + 96);
    if ( (v23 & 2) != 0 )
    {
      if ( (v23 & 1) == 0 || (v24 = *(_QWORD *)(v21 + 112), *(_QWORD *)v24) )
      {
        if ( (unsigned int)sub_87DE40(v19[2], v18) )
        {
LABEL_52:
          v22 = 1;
          goto LABEL_33;
        }
        goto LABEL_30;
      }
    }
    else
    {
      v24 = *(_QWORD *)(v21 + 112);
    }
    if ( !*(_BYTE *)(v24 + 25) || (unsigned int)sub_87D890(v18) )
      goto LABEL_52;
    if ( *(_BYTE *)(*(_QWORD *)(v21 + 112) + 25LL) != 1 )
    {
LABEL_31:
      if ( a10 )
      {
        a3 = 0;
        v22 = 1;
        *a10 = 1;
      }
      else
      {
        v22 = 1;
        a3 = sub_6E53E0(7, 0x10Du, a9);
        if ( a3 )
        {
          sub_685260(7u, 0x10Du, a9, *(_QWORD *)(v21 + 40));
          a3 = 0;
        }
      }
      while ( 1 )
      {
LABEL_33:
        v18 = *(_QWORD *)(v21 + 40);
        v25 = 0;
        if ( (v10[35] & 0xFB) == 8 )
          v25 = (unsigned int)sub_8D4C10(v10, dword_4F077C4 != 2);
        for ( i = v18; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v27 = sub_73C570(i, v25, -1);
        v29 = v27;
        if ( v36 )
        {
          v29 = sub_72D740(v27, v33, v36, v28, v27);
          v30 = (__int64 *)*a8;
        }
        else
        {
          v30 = (__int64 *)*a8;
          if ( (*(_BYTE *)(*a8 + 25) & 3) == 0 )
          {
            v41 = v27;
            v32 = sub_731410(*a8, 1);
            v29 = v41;
            v30 = (__int64 *)v32;
          }
        }
        v40 = sub_73DBF0(14, v29, v30);
        *a8 = v40;
        sub_730580(v30, v40);
        result = v40;
        if ( a6 )
        {
          *(_BYTE *)(v40 + 27) |= 2u;
          *(_QWORD *)(v40 + 28) = *(_QWORD *)a9;
          *(_BYTE *)(v40 + 58) = v35 | *(_BYTE *)(v40 + 58) & 0xBF;
        }
        else
        {
          v31 = v35 | *(_BYTE *)(v40 + 58) & 0xBF;
          *(_BYTE *)(v40 + 58) = v31;
          if ( v22 )
          {
            if ( v19 != v39 )
              *(_BYTE *)(v40 + 58) = v31 | 0x80;
            goto LABEL_43;
          }
        }
        if ( !v19 )
          return result;
LABEL_43:
        result = (__int64)v39;
        v19 = (_QWORD *)*v19;
        if ( (_QWORD *)*v39 == v19 )
          return result;
        if ( v19 )
          goto LABEL_25;
LABEL_45:
        v21 = a1;
        v22 = 0;
      }
    }
    v22 = 1;
    if ( (unsigned int)sub_87D970(v18) )
      goto LABEL_33;
LABEL_30:
    if ( *(_BYTE *)(*(_QWORD *)(v21 + 112) + 25LL) == 1
      && dword_4F077BC
      && qword_4F077A8 <= 0x9DCFu
      && (unsigned int)sub_87E070(v21, a1) )
    {
      goto LABEL_52;
    }
    goto LABEL_31;
  }
  if ( a5 )
  {
    if ( a4 )
      sub_685330(0x11Eu, a9, *(_QWORD *)(a1 + 40));
    goto LABEL_16;
  }
  if ( !a4 )
    goto LABEL_16;
  if ( a10 )
  {
    *a10 = 1;
  }
  else if ( (unsigned int)sub_6E5430() )
  {
    a2 = a9;
    v14 = 286;
    sub_685360(0x11Eu, a9, *(_QWORD *)(a1 + 40));
  }
  result = sub_7305B0(v14, a2);
  *a8 = result;
  return result;
}
