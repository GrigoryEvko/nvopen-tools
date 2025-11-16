// Function: sub_18A22B0
// Address: 0x18a22b0
//
__int64 __fastcall sub_18A22B0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v10; // r13
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 v17; // r14
  __int64 *v18; // rsi
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // r12
  _QWORD *v24; // rdi
  double v25; // xmm4_8
  double v26; // xmm5_8
  bool v28; // r14
  __int64 v29; // r12
  double v30; // xmm4_8
  double v31; // xmm5_8
  __int64 v32; // r15
  _QWORD *v34; // [rsp+8h] [rbp-78h]
  unsigned __int8 i; // [rsp+18h] [rbp-68h]
  __int64 v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+28h] [rbp-58h]
  _QWORD v39[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v40; // [rsp+40h] [rbp-40h]

  v38 = *(_QWORD *)(a1 + 80);
  v37 = a1 + 72;
  for ( i = 0; v37 != v38; v38 = *(_QWORD *)(v38 + 8) )
  {
    v10 = (_QWORD *)(v38 - 24);
    if ( !v38 )
      v10 = 0;
    v11 = sub_157EBA0((__int64)v10);
    v12 = v11;
    if ( *(_BYTE *)(v11 + 16) == 29 )
    {
      if ( (unsigned __int8)sub_1560260((_QWORD *)(v11 + 56), -1, 30)
        || (v13 = *(_QWORD *)(v12 - 72), !*(_BYTE *)(v13 + 16))
        && (v39[0] = *(_QWORD *)(v13 + 112), (unsigned __int8)sub_1560260(v39, -1, 30)) )
      {
        v28 = sub_14DDD80(a1);
        if ( v28 )
        {
          v29 = *(_QWORD *)(v12 - 24);
          sub_1AF0970(v10, 0);
          v32 = *(_QWORD *)(v29 + 8);
          if ( v32 )
          {
            while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v32) + 16) - 25) > 9u )
            {
              v32 = *(_QWORD *)(v32 + 8);
              if ( !v32 )
                goto LABEL_27;
            }
            i = v28;
          }
          else
          {
LABEL_27:
            sub_18A1FF0(v29, a2, a3, a4, a5, a6, v30, v31, a9, a10);
            i = v28;
          }
        }
      }
    }
    v14 = (__int64 *)v10[6];
    if ( v10 + 5 != v14 )
    {
      while ( 1 )
      {
        v16 = v14;
        v14 = (__int64 *)v14[1];
        if ( *((_BYTE *)v16 - 8) == 78 )
        {
          if ( (unsigned __int8)sub_1560260(v16 + 4, -1, 29)
            || (v15 = *(v16 - 6), !*(_BYTE *)(v15 + 16))
            && (v39[0] = *(_QWORD *)(v15 + 112), (unsigned __int8)sub_1560260(v39, -1, 29)) )
          {
            if ( !v14 )
              BUG();
            if ( *((_BYTE *)v14 - 8) != 31 )
              break;
          }
        }
        if ( v10 + 5 == v14 )
          goto LABEL_20;
      }
      v40 = 257;
      v17 = sub_157FBF0(v10, v14, (__int64)v39);
      v34 = (_QWORD *)(v10[5] & 0xFFFFFFFFFFFFFFF8LL);
      sub_157EA20((__int64)(v10 + 5), (__int64)(v34 - 3));
      v18 = (__int64 *)v34[1];
      v19 = *v34 & 0xFFFFFFFFFFFFFFF8LL;
      v20 = v19 | *v18 & 7;
      *v18 = v20;
      *(_QWORD *)(v19 + 8) = v18;
      *v34 &= 7uLL;
      v34[1] = 0;
      sub_164BEC0((__int64)(v34 - 3), (__int64)v18, v20, v19, a3, a4, a5, a6, v21, v22, a9, a10);
      v23 = sub_157E9C0((__int64)v10);
      v24 = sub_1648A60(56, 0);
      if ( v24 )
        sub_15F82E0((__int64)v24, v23, (__int64)v10);
      sub_18A1FF0(v17, a2, a3, a4, a5, a6, v25, v26, a9, a10);
      i = 1;
    }
LABEL_20:
    ;
  }
  return i;
}
