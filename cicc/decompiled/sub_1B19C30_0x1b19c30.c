// Function: sub_1B19C30
// Address: 0x1b19c30
//
unsigned __int8 *__fastcall sub_1B19C30(
        __int64 *a1,
        unsigned __int8 *a2,
        __int64 a3,
        unsigned int a4,
        int a5,
        double a6,
        double a7,
        double a8,
        __int64 a9,
        __int64 *a10,
        __int64 a11)
{
  __int64 v11; // rax
  unsigned __int8 *v14; // r12
  __int64 v15; // rbx
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // r15
  _QWORD *v21; // rax
  __int64 v22; // rdi
  unsigned __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rdx
  unsigned __int8 *v28; // rsi
  __int64 v30; // [rsp+20h] [rbp-A0h]
  unsigned __int64 *v31; // [rsp+20h] [rbp-A0h]
  unsigned int v33; // [rsp+2Ch] [rbp-94h]
  __int64 v34; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v35; // [rsp+48h] [rbp-78h] BYREF
  __int64 v36; // [rsp+50h] [rbp-70h] BYREF
  __int16 v37; // [rsp+60h] [rbp-60h]
  __int64 v38[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v39; // [rsp+80h] [rbp-40h]

  v11 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
  if ( (_DWORD)v11 )
  {
    v33 = a4 - 51;
    v14 = a2;
    v15 = 0;
    v34 = (unsigned int)v11;
    while ( 1 )
    {
      v16 = (_QWORD *)a1[3];
      v37 = 257;
      v17 = sub_1643350(v16);
      v18 = sub_159C470(v17, v15, 0);
      if ( *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(v18 + 16) > 0x10u )
      {
        v30 = v18;
        v39 = 257;
        v21 = sub_1648A60(56, 2u);
        v19 = v21;
        if ( v21 )
          sub_15FA320((__int64)v21, (_QWORD *)a3, v30, (__int64)v38, 0);
        v22 = a1[1];
        if ( v22 )
        {
          v31 = (unsigned __int64 *)a1[2];
          sub_157E9D0(v22 + 40, (__int64)v19);
          v23 = *v31;
          v24 = v19[3] & 7LL;
          v19[4] = v31;
          v23 &= 0xFFFFFFFFFFFFFFF8LL;
          v19[3] = v23 | v24;
          *(_QWORD *)(v23 + 8) = v19 + 3;
          *v31 = *v31 & 7 | (unsigned __int64)(v19 + 3);
        }
        sub_164B780((__int64)v19, &v36);
        v25 = *a1;
        if ( *a1 )
        {
          v35 = (unsigned __int8 *)*a1;
          sub_1623A60((__int64)&v35, v25, 2);
          v26 = v19[6];
          v27 = (__int64)(v19 + 6);
          if ( v26 )
          {
            sub_161E7C0((__int64)(v19 + 6), v26);
            v27 = (__int64)(v19 + 6);
          }
          v28 = v35;
          v19[6] = v35;
          if ( v28 )
            sub_1623210((__int64)&v35, v28, v27);
        }
      }
      else
      {
        v19 = (_QWORD *)sub_15A37D0((_BYTE *)a3, v18, 0);
      }
      if ( v33 > 1 )
      {
        v38[0] = (__int64)"bin.rdx";
        v39 = 259;
        v14 = (unsigned __int8 *)sub_1904E90((__int64)a1, a4, (__int64)v14, (__int64)v19, v38, 0, a6, a7, a8);
        if ( !a11 )
          goto LABEL_4;
LABEL_10:
        ++v15;
        sub_1B188F0(v14, a10, a11, 0);
        if ( v34 == v15 )
          return v14;
      }
      else
      {
        v14 = (unsigned __int8 *)sub_1B16290((__int64)a1, a5, v14, (__int64)v19);
        if ( a11 )
          goto LABEL_10;
LABEL_4:
        if ( v34 == ++v15 )
          return v14;
      }
    }
  }
  return a2;
}
