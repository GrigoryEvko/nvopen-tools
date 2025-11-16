// Function: sub_20D7260
// Address: 0x20d7260
//
__int64 __fastcall sub_20D7260(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // rax
  __int64 *v5; // r13
  _QWORD *v6; // r15
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  _QWORD *v10; // r14
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int16 *v15; // r12
  __int64 v16; // r15
  unsigned __int16 *v17; // r12
  unsigned int v18; // r15d
  unsigned __int64 v19; // rax
  __int64 v21; // [rsp+20h] [rbp-90h]
  __int64 v22; // [rsp+28h] [rbp-88h]
  unsigned __int16 *v23; // [rsp+30h] [rbp-80h]
  _QWORD *v24; // [rsp+38h] [rbp-78h]
  __int64 v25; // [rsp+48h] [rbp-68h] BYREF
  __m128i v26; // [rsp+50h] [rbp-60h] BYREF
  __int64 v27; // [rsp+60h] [rbp-50h]
  __int64 v28; // [rsp+68h] [rbp-48h]
  __int64 v29; // [rsp+70h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 139) )
  {
    v4 = (_QWORD *)a2[3];
    v5 = (__int64 *)(a1 + 184);
    *(_DWORD *)(a1 + 200) = 0;
    v6 = v4;
    v24 = v4;
    sub_1DC2AE0((__int64 *)(a1 + 184), v4);
    v10 = v6 + 3;
    do
    {
      v11 = (__int64 *)(*v10 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = (__int64)v11;
      if ( !v11 )
        BUG();
      v10 = (_QWORD *)(*v10 & 0xFFFFFFFFFFFFFFF8LL);
      v13 = *v11;
      if ( (v13 & 4) == 0 && (*(_BYTE *)(v12 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
          v10 = (_QWORD *)v14;
          if ( (*(_BYTE *)(v14 + 46) & 4) == 0 )
            break;
          v13 = *(_QWORD *)v14;
        }
      }
      sub_1DC2260(v5, (unsigned __int64)v10, v12, v7, v8, v9);
    }
    while ( a2 != v10 );
    v15 = *(unsigned __int16 **)(a3 + 160);
    v23 = v15;
    v16 = sub_1DD77D0(a3);
    if ( v15 != (unsigned __int16 *)v16 )
    {
      v17 = (unsigned __int16 *)v16;
      do
      {
        v18 = *v17;
        if ( (unsigned __int8)sub_1DC24A0(v5, *(_QWORD *)(a1 + 152), v18) )
        {
          v25 = 0;
          v21 = v24[7];
          v22 = (__int64)sub_1E0B640(v21, *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL) + 576LL, &v25, 0);
          sub_1DD5BA0(v24 + 2, v22);
          v19 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v22 + 8) = v10;
          *(_QWORD *)v22 = v19 | *(_QWORD *)v22 & 7LL;
          *(_QWORD *)(v19 + 8) = v22;
          *v10 = v22 | *v10 & 7LL;
          v26.m128i_i64[0] = 0x10000000;
          v27 = 0;
          v26.m128i_i32[2] = v18;
          v28 = 0;
          v29 = 0;
          sub_1E1A9C0(v22, v21, &v26);
          if ( v25 )
            sub_161E7C0((__int64)&v25, v25);
        }
        v17 += 4;
      }
      while ( v23 != v17 );
    }
  }
  return (*(__int64 (__fastcall **)(_QWORD, _QWORD *, __int64))(**(_QWORD **)(a1 + 144) + 312LL))(
           *(_QWORD *)(a1 + 144),
           a2,
           a3);
}
