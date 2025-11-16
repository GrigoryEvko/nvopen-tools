// Function: sub_2D1DCF0
// Address: 0x2d1dcf0
//
unsigned __int64 *__fastcall sub_2D1DCF0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        _BYTE *a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8,
        _QWORD *a9,
        _QWORD *a10)
{
  unsigned int v10; // eax
  __int64 v11; // r12
  __int64 v12; // rbx
  _QWORD *v13; // r15
  __int64 v14; // r14
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  __int64 v18; // r8
  _QWORD *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rcx
  _QWORD *v22; // rdi
  unsigned __int64 *v23; // rsi
  __int64 v24; // rbx
  unsigned __int64 v25; // r12
  unsigned __int64 *v26; // r13
  __int64 v27; // rdx
  __int64 v29; // rax
  __int64 v30; // r9
  _QWORD *v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // r9
  _QWORD *v34; // rdx
  __int64 v35; // rcx
  unsigned __int64 *v36; // rdi
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // rsi
  _QWORD *v40; // rax
  __int64 v41; // [rsp+0h] [rbp-A0h]
  __int64 v42; // [rsp+8h] [rbp-98h]
  __int64 v43; // [rsp+10h] [rbp-90h]
  _BYTE *v44; // [rsp+18h] [rbp-88h]
  __int64 v45; // [rsp+18h] [rbp-88h]
  int v46; // [rsp+18h] [rbp-88h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  _QWORD *v50; // [rsp+30h] [rbp-70h]
  __int64 v52; // [rsp+40h] [rbp-60h]
  unsigned __int64 *v53; // [rsp+48h] [rbp-58h] BYREF
  unsigned __int64 v54; // [rsp+58h] [rbp-48h] BYREF
  unsigned __int64 v55; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 *v56[7]; // [rsp+68h] [rbp-38h] BYREF

  v52 = a1 + 112;
  v10 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v53 = a3;
  v50 = a9 + 1;
  if ( v10 )
  {
    v11 = a2;
    v12 = 0;
    v13 = a8 + 1;
    v14 = 32LL * v10;
    do
    {
      if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
        v16 = *(_QWORD *)(v11 - 8);
      else
        v16 = v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
      v17 = *(_QWORD *)(v16 + v12);
      if ( *(_BYTE *)v17 <= 0x1Cu )
        goto LABEL_6;
      v54 = v17;
      if ( a6 != *(_QWORD *)(v17 + 40) || *(_BYTE *)v17 == 84 )
        goto LABEL_6;
      v18 = (__int64)(a8 + 1);
      v19 = (_QWORD *)a8[2];
      if ( !v19 )
        goto LABEL_17;
      do
      {
        while ( 1 )
        {
          v20 = v19[2];
          v21 = v19[3];
          if ( v19[4] >= v17 )
            break;
          v19 = (_QWORD *)v19[3];
          if ( !v21 )
            goto LABEL_15;
        }
        v18 = (__int64)v19;
        v19 = (_QWORD *)v19[2];
      }
      while ( v20 );
LABEL_15:
      if ( v13 == (_QWORD *)v18 || *(_QWORD *)(v18 + 32) > v17 )
      {
LABEL_17:
        v56[0] = &v54;
        v18 = sub_2D1BF10(a8, v18, v56);
      }
      if ( *(_QWORD *)(v18 + 40) )
        goto LABEL_6;
      v44 = (_BYTE *)v54;
      if ( (unsigned __int8)sub_B46490(v54) || !sub_2D1D770(a1, v44, (unsigned __int64)v53, a7, a9, a10) )
        goto LABEL_6;
      v22 = (_QWORD *)v54;
      v23 = v53;
      if ( *(_QWORD *)(v54 + 16) )
      {
        v45 = v12;
        v24 = *(_QWORD *)(v54 + 16);
        v43 = v11;
        v25 = v53[5];
        v42 = a6;
        v26 = v53;
        do
        {
          v27 = *(_QWORD *)(v24 + 24);
          if ( *(_BYTE *)v27 > 0x1Cu
            && v26 != (unsigned __int64 *)v27
            && v25 == *(_QWORD *)(v27 + 40)
            && !(unsigned __int8)sub_2D1CFB0(v52, (__int64)v26, v27) )
          {
            v12 = v45;
            v11 = v43;
            a6 = v42;
            goto LABEL_6;
          }
          v24 = *(_QWORD *)(v24 + 8);
        }
        while ( v24 );
        v12 = v45;
        v11 = v43;
        a6 = v42;
        v22 = (_QWORD *)v54;
        v23 = v53;
      }
      v29 = v41;
      LOWORD(v29) = 0;
      v41 = v29;
      sub_B444E0(v22, (__int64)(v23 + 3), v29);
      v55 = v54;
      v56[0] = v53;
      v46 = *(_DWORD *)sub_DA65E0(v52, (__int64 *)v56);
      *(_DWORD *)sub_DA65E0(v52, (__int64 *)&v55) = v46;
      if ( v54 == a5 )
      {
        v39 = (__int64)(a8 + 1);
        v40 = (_QWORD *)a8[2];
        if ( !v40 )
          goto LABEL_71;
        do
        {
          if ( v40[4] < a5 )
          {
            v40 = (_QWORD *)v40[3];
          }
          else
          {
            v39 = (__int64)v40;
            v40 = (_QWORD *)v40[2];
          }
        }
        while ( v40 );
        if ( v13 == (_QWORD *)v39 || *(_QWORD *)(v39 + 32) > a5 )
        {
LABEL_71:
          v56[0] = &v54;
          v39 = sub_2D1BF10(a8, v39, v56);
        }
        *(_QWORD *)(v39 + 40) = 0;
      }
      else
      {
        v30 = (__int64)(a8 + 1);
        v31 = (_QWORD *)a8[2];
        if ( !v31 )
          goto LABEL_33;
        do
        {
          if ( v31[4] < v54 )
          {
            v31 = (_QWORD *)v31[3];
          }
          else
          {
            v30 = (__int64)v31;
            v31 = (_QWORD *)v31[2];
          }
        }
        while ( v31 );
        if ( v13 == (_QWORD *)v30 || *(_QWORD *)(v30 + 32) > v54 )
        {
LABEL_33:
          v56[0] = &v54;
          v30 = sub_2D1BF10(a8, v30, v56);
        }
        *(_QWORD *)(v30 + 40) = a5;
      }
      v32 = (_QWORD *)a9[2];
      if ( v32 )
      {
        v33 = (__int64)(a9 + 1);
        v34 = (_QWORD *)a9[2];
        do
        {
          if ( v34[4] < (unsigned __int64)v53 )
          {
            v34 = (_QWORD *)v34[3];
          }
          else
          {
            v33 = (__int64)v34;
            v34 = (_QWORD *)v34[2];
          }
        }
        while ( v34 );
        if ( (_QWORD *)v33 != v50 && *(_QWORD *)(v33 + 32) <= (unsigned __int64)v53 )
        {
          v35 = *(_QWORD *)(v33 + 40);
LABEL_48:
          v36 = (unsigned __int64 *)v54;
          v37 = (__int64)(a9 + 1);
          do
          {
            if ( v32[4] < v54 )
            {
              v32 = (_QWORD *)v32[3];
            }
            else
            {
              v37 = (__int64)v32;
              v32 = (_QWORD *)v32[2];
            }
          }
          while ( v32 );
          if ( (_QWORD *)v37 != v50 && *(_QWORD *)(v37 + 32) <= v54 )
            goto LABEL_56;
          goto LABEL_55;
        }
      }
      else
      {
        v33 = (__int64)(a9 + 1);
      }
      v56[0] = (unsigned __int64 *)&v53;
      v35 = *(_QWORD *)(sub_2D1BF10(a9, v33, v56) + 40);
      v32 = (_QWORD *)a9[2];
      if ( v32 )
        goto LABEL_48;
      v37 = (__int64)(a9 + 1);
LABEL_55:
      v47 = v35;
      v56[0] = &v54;
      v38 = sub_2D1BF10(a9, v37, v56);
      v36 = (unsigned __int64 *)v54;
      v35 = v47;
      v37 = v38;
LABEL_56:
      *(_QWORD *)(v37 + 40) = v35;
      v53 = v36;
      *a4 = 1;
LABEL_6:
      v12 += 32;
    }
    while ( v14 != v12 );
  }
  return v53;
}
