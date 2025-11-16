// Function: sub_1804D00
// Address: 0x1804d00
//
void __fastcall sub_1804D00(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 v7; // rbx
  int v8; // edx
  __int64 v9; // rax
  char v10; // r12
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned int v14; // r13d
  __int64 v15; // r14
  unsigned __int64 i; // rax
  char v17; // cl
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 **v21; // rax
  __int64 v22; // r14
  __int64 **v23; // rax
  __int64 v24; // r9
  _QWORD *v25; // rax
  _QWORD *v26; // r13
  __int64 v27; // rax
  unsigned __int64 *v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 *v36; // r13
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // r13
  unsigned __int8 *v42; // rsi
  unsigned __int64 v44; // [rsp+18h] [rbp-B8h]
  __int64 v45; // [rsp+28h] [rbp-A8h]
  __int64 v46; // [rsp+28h] [rbp-A8h]
  __int64 v47; // [rsp+28h] [rbp-A8h]
  __int64 v48; // [rsp+28h] [rbp-A8h]
  __int64 v49; // [rsp+28h] [rbp-A8h]
  __int64 v53; // [rsp+58h] [rbp-78h] BYREF
  __int64 v54; // [rsp+60h] [rbp-70h] BYREF
  __int16 v55; // [rsp+70h] [rbp-60h]
  __int64 v56[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v57; // [rsp+90h] [rbp-40h]

  if ( a4 < a5 )
  {
    v7 = a4;
    v8 = *(_DWORD *)(a1[1] + 224LL);
    v9 = 8;
    if ( (unsigned __int64)(v8 / 8) < 8 )
      v9 = v8 / 8;
    v44 = v9;
    v10 = *(_BYTE *)sub_1632FA0(*(_QWORD *)(*a1 + 40LL));
    do
    {
      while ( !*(_BYTE *)(a2 + v7) )
      {
        if ( a5 <= ++v7 )
          return;
      }
      v11 = v44;
      if ( v44 <= a5 - v7 )
      {
        v11 = v44;
        v12 = v44 - 1;
        if ( v44 == 1 )
          goto LABEL_41;
      }
      else
      {
        do
          v11 >>= 1;
        while ( v11 > a5 - v7 );
        v12 = v11 - 1;
        if ( v11 == 1 )
        {
LABEL_41:
          v14 = 8;
          v11 = 1;
LABEL_14:
          v15 = 0;
          for ( i = 0; i < v11; ++i )
          {
            while ( 1 )
            {
              v18 = *(unsigned __int8 *)(a3 + v7 + i);
              if ( v10 )
                break;
              v17 = 8 * i++;
              v15 |= v18 << v17;
              if ( i >= v11 )
                goto LABEL_18;
            }
            v15 = v18 | (v15 << 8);
          }
          goto LABEL_18;
        }
      }
      do
      {
        if ( *(_BYTE *)(a2 + v7 + v12) )
          break;
        do
        {
          v13 = v11;
          v11 >>= 1;
        }
        while ( v11 >= v12 );
        v11 = v13;
        --v12;
      }
      while ( v12 );
      v14 = 8 * v11;
      if ( v11 )
        goto LABEL_14;
      v15 = 0;
LABEL_18:
      v57 = 257;
      v19 = sub_15A0680(a1[61], v7, 0);
      v45 = sub_12899C0((__int64 *)a6, a7, v19, (__int64)v56, 0, 0);
      v20 = sub_1644C60(*(_QWORD **)(a6 + 24), v14);
      v21 = (__int64 **)sub_159C470(v20, v15, 0);
      v55 = 257;
      v22 = (__int64)v21;
      v23 = (__int64 **)sub_1647190(*v21, 0);
      v24 = v45;
      if ( v23 != *(__int64 ***)v45 )
      {
        if ( *(_BYTE *)(v45 + 16) > 0x10u )
        {
          v57 = 257;
          v34 = sub_15FDBD0(46, v45, (__int64)v23, (__int64)v56, 0);
          v35 = *(_QWORD *)(a6 + 8);
          if ( v35 )
          {
            v36 = *(__int64 **)(a6 + 16);
            v47 = v34;
            sub_157E9D0(v35 + 40, v34);
            v34 = v47;
            v37 = *v36;
            v38 = *(_QWORD *)(v47 + 24);
            *(_QWORD *)(v47 + 32) = v36;
            v37 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v47 + 24) = v37 | v38 & 7;
            *(_QWORD *)(v37 + 8) = v47 + 24;
            *v36 = *v36 & 7 | (v47 + 24);
          }
          v48 = v34;
          sub_164B780(v34, &v54);
          v24 = v48;
          v39 = *(_QWORD *)a6;
          if ( *(_QWORD *)a6 )
          {
            v53 = *(_QWORD *)a6;
            sub_1623A60((__int64)&v53, v39, 2);
            v24 = v48;
            v40 = *(_QWORD *)(v48 + 48);
            v41 = v48 + 48;
            if ( v40 )
            {
              sub_161E7C0(v48 + 48, v40);
              v24 = v48;
            }
            v42 = (unsigned __int8 *)v53;
            *(_QWORD *)(v24 + 48) = v53;
            if ( v42 )
            {
              v49 = v24;
              sub_1623210((__int64)&v53, v42, v41);
              v24 = v49;
            }
          }
        }
        else
        {
          v24 = sub_15A46C0(46, (__int64 ***)v45, v23, 0);
        }
      }
      v46 = v24;
      v57 = 257;
      v25 = sub_1648A60(64, 2u);
      v26 = v25;
      if ( v25 )
        sub_15F9650((__int64)v25, v22, v46, 0, 0);
      v27 = *(_QWORD *)(a6 + 8);
      if ( v27 )
      {
        v28 = *(unsigned __int64 **)(a6 + 16);
        sub_157E9D0(v27 + 40, (__int64)v26);
        v29 = v26[3];
        v30 = *v28;
        v26[4] = v28;
        v30 &= 0xFFFFFFFFFFFFFFF8LL;
        v26[3] = v30 | v29 & 7;
        *(_QWORD *)(v30 + 8) = v26 + 3;
        *v28 = *v28 & 7 | (unsigned __int64)(v26 + 3);
      }
      sub_164B780((__int64)v26, v56);
      v31 = *(_QWORD *)a6;
      if ( *(_QWORD *)a6 )
      {
        v53 = *(_QWORD *)a6;
        sub_1623A60((__int64)&v53, v31, 2);
        v32 = v26[6];
        if ( v32 )
          sub_161E7C0((__int64)(v26 + 6), v32);
        v33 = (unsigned __int8 *)v53;
        v26[6] = v53;
        if ( v33 )
          sub_1623210((__int64)&v53, v33, (__int64)(v26 + 6));
      }
      v7 += v11;
      sub_15F9450((__int64)v26, 1u);
    }
    while ( a5 > v7 );
  }
}
