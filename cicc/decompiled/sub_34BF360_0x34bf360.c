// Function: sub_34BF360
// Address: 0x34bf360
//
__int64 __fastcall sub_34BF360(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r15
  _QWORD *v6; // r15
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned int *v11; // r13
  unsigned int *v12; // rbx
  __int32 v13; // r13d
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // r14
  _QWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r8
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  _QWORD *v24; // [rsp+18h] [rbp-B8h]
  __int64 v25; // [rsp+18h] [rbp-B8h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  unsigned int *v27; // [rsp+28h] [rbp-A8h]
  __int64 v28; // [rsp+30h] [rbp-A0h]
  __int64 v29; // [rsp+40h] [rbp-90h] BYREF
  __int64 v30; // [rsp+48h] [rbp-88h] BYREF
  __int64 v31; // [rsp+50h] [rbp-80h] BYREF
  __int64 v32; // [rsp+58h] [rbp-78h]
  __int64 v33; // [rsp+60h] [rbp-70h]
  __m128i v34; // [rsp+70h] [rbp-60h] BYREF
  __int64 v35; // [rsp+80h] [rbp-50h]
  __int64 v36; // [rsp+88h] [rbp-48h]
  __int64 v37; // [rsp+90h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 131) )
  {
    v4 = a2[3];
    *(_QWORD *)(a1 + 184) = 0;
    v5 = v4;
    v28 = v4;
    sub_3508720(a1 + 168, v4);
    v6 = (_QWORD *)(v5 + 48);
    do
    {
      v7 = (_QWORD *)(*v6 & 0xFFFFFFFFFFFFFFF8LL);
      v8 = v7;
      if ( !v7 )
        BUG();
      v6 = (_QWORD *)(*v6 & 0xFFFFFFFFFFFFFFF8LL);
      v9 = *v7;
      if ( (v9 & 4) == 0 && (*((_BYTE *)v8 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          v6 = (_QWORD *)v10;
          if ( (*(_BYTE *)(v10 + 44) & 4) == 0 )
            break;
          v9 = *(_QWORD *)v10;
        }
      }
      sub_3508F10(a1 + 168, v6, v8);
    }
    while ( a2 != v6 );
    v11 = *(unsigned int **)(a3 + 192);
    v27 = v11;
    v12 = (unsigned int *)sub_2E33140(a3);
    if ( v11 != v12 )
    {
      do
      {
        v13 = *v12;
        if ( (unsigned __int8)sub_35080D0(a1 + 168, *(_QWORD *)(a1 + 144), *v12) )
        {
          v14 = *(_QWORD *)(a1 + 136);
          v29 = 0;
          v30 = 0;
          v15 = *(_QWORD *)(v14 + 8);
          v31 = 0;
          v32 = 0;
          v16 = *(_QWORD **)(v28 + 32);
          v33 = 0;
          v34.m128i_i64[0] = 0;
          v17 = sub_2E7B380(v16, v15 - 400, (unsigned __int8 **)&v34, 0);
          v18 = (__int64)v17;
          if ( v34.m128i_i64[0] )
          {
            v24 = v17;
            sub_B91220((__int64)&v34, v34.m128i_i64[0]);
            v18 = (__int64)v24;
          }
          v25 = v18;
          sub_2E31040((__int64 *)(v28 + 40), v18);
          v19 = v25;
          v20 = *(_QWORD *)v25;
          v21 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v25 + 8) = v6;
          *(_QWORD *)v25 = v21 | v20 & 7;
          *(_QWORD *)(v21 + 8) = v25;
          *v6 = v25 | *v6 & 7LL;
          if ( v32 )
          {
            sub_2E882B0(v25, (__int64)v16, v32);
            v19 = v25;
          }
          if ( v33 )
          {
            v26 = v19;
            sub_2E88680(v19, (__int64)v16, v33);
            v19 = v26;
          }
          v34.m128i_i64[0] = 0x10000000;
          v35 = 0;
          v34.m128i_i32[2] = v13;
          v36 = 0;
          v37 = 0;
          sub_2E8EAD0(v19, (__int64)v16, &v34);
          if ( v31 )
            sub_B91220((__int64)&v31, v31);
          if ( v30 )
            sub_B91220((__int64)&v30, v30);
          if ( v29 )
            sub_B91220((__int64)&v29, v29);
        }
        v12 += 6;
      }
      while ( v27 != v12 );
    }
  }
  return (*(__int64 (__fastcall **)(_QWORD, _QWORD *, __int64))(**(_QWORD **)(a1 + 136) + 400LL))(
           *(_QWORD *)(a1 + 136),
           a2,
           a3);
}
