// Function: sub_17D3020
// Address: 0x17d3020
//
unsigned __int64 __fastcall sub_17D3020(
        _QWORD *a1,
        __int64 *a2,
        __int64 a3,
        _BYTE *a4,
        unsigned int a5,
        unsigned int a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v11; // r14
  unsigned __int64 result; // rax
  unsigned int v13; // ebx
  _QWORD *v14; // rax
  _QWORD *v15; // r15
  __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int8 *v22; // rsi
  unsigned int v23; // esi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // ecx
  __int64 v33; // rdx
  unsigned int v34; // r13d
  unsigned int v35; // r15d
  __int64 v36; // rax
  _QWORD *v37; // rax
  int v38; // ebx
  unsigned __int64 v39; // [rsp+8h] [rbp-A8h]
  _BYTE *v40; // [rsp+10h] [rbp-A0h]
  __int64 v41; // [rsp+18h] [rbp-98h]
  unsigned int v43; // [rsp+24h] [rbp-8Ch]
  unsigned int v44; // [rsp+24h] [rbp-8Ch]
  unsigned int v45; // [rsp+24h] [rbp-8Ch]
  unsigned int v48; // [rsp+38h] [rbp-78h]
  __int64 v49; // [rsp+38h] [rbp-78h]
  unsigned __int64 *v50; // [rsp+38h] [rbp-78h]
  __int64 v51[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v52; // [rsp+50h] [rbp-60h]
  __int64 v53[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v54; // [rsp+70h] [rbp-40h]

  v11 = sub_1632FA0(*(_QWORD *)(*a1 + 40LL));
  v48 = sub_15A9FE0(v11, *(_QWORD *)(a1[1] + 176LL));
  result = sub_127FA20(v11, *(_QWORD *)(a1[1] + 176LL)) + 7;
  v39 = result >> 3;
  v43 = result >> 3;
  if ( v48 > a6 || (unsigned int)(result >> 3) <= 4 )
    goto LABEL_3;
  v25 = sub_1632FA0(*(_QWORD *)(*a1 + 40LL));
  if ( (unsigned int)((unsigned __int64)(sub_127FA20(v25, *(_QWORD *)(a1[1] + 176LL)) + 7) >> 3) == 4 )
  {
    v41 = a3;
  }
  else
  {
    v26 = a1[1];
    v54 = 257;
    v27 = sub_17CE200(a2, a3, *(__int64 ***)(v26 + 176), 0, v53);
    v54 = 257;
    v52 = 257;
    v28 = sub_15A0680(*(_QWORD *)v27, 32, 0);
    if ( *(_BYTE *)(v27 + 16) > 0x10u || *(_BYTE *)(v28 + 16) > 0x10u )
      v29 = (__int64)sub_17D2EF0(a2, 23, (__int64 *)v27, v28, v51, 0, 0);
    else
      v29 = sub_15A2D50((__int64 *)v27, v28, 0, 0, a7, a8, a9);
    v41 = sub_156D390(a2, v27, v29, (__int64)v53);
  }
  v30 = a1[1];
  v54 = 257;
  v31 = sub_1646BA0(*(__int64 **)(v30 + 176), 0);
  v40 = (_BYTE *)sub_12A95D0(a2, (__int64)a4, v31, (__int64)v53);
  v45 = a5 / v43;
  result = a5;
  if ( (unsigned int)v39 > a5 )
  {
LABEL_3:
    v13 = 0;
  }
  else
  {
    v32 = a6;
    v33 = (__int64)v40;
    v34 = 0;
    v35 = v32;
    while ( 1 )
    {
      ++v34;
      v37 = sub_12A8F50(a2, v41, v33, 0);
      sub_15F9450((__int64)v37, v35);
      if ( v34 >= v45 )
        break;
      v36 = a1[1];
      v54 = 257;
      v35 = v48;
      v33 = sub_17CE2F0((__int64)a2, *(_QWORD *)(v36 + 176), v40, v34, v53);
    }
    result = a5;
    v38 = 1;
    a6 = v48;
    if ( (unsigned int)v39 <= a5 )
      v38 = v45;
    v13 = ((unsigned int)v39 >> 2) * v38;
  }
  v44 = (a5 + 3) >> 2;
  if ( v13 < v44 )
  {
    do
    {
      v24 = (__int64)a4;
      if ( v13 )
      {
        v54 = 257;
        v24 = sub_17CE2F0((__int64)a2, 0, a4, v13, v53);
      }
      v49 = v24;
      v54 = 257;
      v14 = sub_1648A60(64, 2u);
      v15 = v14;
      if ( v14 )
        sub_15F9650((__int64)v14, a3, v49, 0, 0);
      v16 = a2[1];
      if ( v16 )
      {
        v50 = (unsigned __int64 *)a2[2];
        sub_157E9D0(v16 + 40, (__int64)v15);
        v17 = *v50;
        v18 = v15[3] & 7LL;
        v15[4] = v50;
        v17 &= 0xFFFFFFFFFFFFFFF8LL;
        v15[3] = v17 | v18;
        *(_QWORD *)(v17 + 8) = v15 + 3;
        *v50 = *v50 & 7 | (unsigned __int64)(v15 + 3);
      }
      sub_164B780((__int64)v15, v53);
      v19 = *a2;
      if ( *a2 )
      {
        v51[0] = *a2;
        sub_1623A60((__int64)v51, v19, 2);
        v20 = v15[6];
        v21 = (__int64)(v15 + 6);
        if ( v20 )
        {
          sub_161E7C0((__int64)(v15 + 6), v20);
          v21 = (__int64)(v15 + 6);
        }
        v22 = (unsigned __int8 *)v51[0];
        v15[6] = v51[0];
        if ( v22 )
          sub_1623210((__int64)v51, v22, v21);
      }
      v23 = a6;
      ++v13;
      a6 = 4;
      result = sub_15F9450((__int64)v15, v23);
    }
    while ( v13 != v44 );
  }
  return result;
}
