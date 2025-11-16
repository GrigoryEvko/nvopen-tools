// Function: sub_2EC6990
// Address: 0x2ec6990
//
void __fastcall sub_2EC6990(__int64 a1)
{
  unsigned __int64 **v2; // r13
  unsigned __int64 *v3; // r14
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdi
  unsigned __int8 i; // [rsp+1Fh] [rbp-D1h] BYREF
  __int64 *v12; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v13; // [rsp+28h] [rbp-C8h]
  _BYTE v14[64]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE *v15; // [rsp+70h] [rbp-80h] BYREF
  __int64 v16; // [rsp+78h] [rbp-78h]
  _BYTE v17[112]; // [rsp+80h] [rbp-70h] BYREF

  sub_2F97F60(a1, *(_QWORD *)(a1 + 3456), 0, 0, 0, 0);
  sub_2EC6470(a1);
  v15 = v17;
  v12 = (__int64 *)v14;
  v13 = 0x800000000LL;
  v16 = 0x800000000LL;
  sub_2EC64C0(a1, (__int64)&v12, (__int64)&v15);
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 3472) + 72LL))(*(_QWORD *)(a1 + 3472), a1);
  sub_2EC65D0(a1, v12, (unsigned int)v13, (__int64)v15, (unsigned int)v16);
  for ( i = 0; ; sub_2EC66C0(a1, (__int64)v2, i) )
  {
    v2 = (unsigned __int64 **)(*(__int64 (__fastcall **)(_QWORD, unsigned __int8 *))(**(_QWORD **)(a1 + 3472) + 104LL))(
                                *(_QWORD *)(a1 + 3472),
                                &i);
    if ( !v2 || !(unsigned __int8)sub_2EC6460(a1) )
      break;
    v3 = *v2;
    v4 = *(_QWORD *)(a1 + 3504);
    if ( i )
    {
      if ( v3 == (unsigned __int64 *)v4 )
      {
        v8 = *(_QWORD *)(a1 + 3512);
        if ( !v4 )
          BUG();
        if ( (*(_BYTE *)v4 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
            v4 = *(_QWORD *)(v4 + 8);
        }
        v9 = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(a1 + 3504) = v9;
        *(_QWORD *)(a1 + 3504) = sub_2EC2050(v9, v8);
      }
      else
      {
        sub_2EC62F0((_QWORD *)a1, *v2, (unsigned __int64 *)v4);
      }
    }
    else
    {
      v5 = sub_2EC1A40(*(_QWORD *)(a1 + 3512), v4);
      v6 = v5;
      if ( (unsigned __int64 *)v5 == v3 )
      {
        *(_QWORD *)(a1 + 3512) = v5;
      }
      else
      {
        v7 = *(_QWORD *)(a1 + 3504);
        if ( v3 == (unsigned __int64 *)v7 )
        {
          if ( !v3 )
            BUG();
          if ( (*(_BYTE *)v3 & 4) == 0 && (*((_BYTE *)v3 + 44) & 8) != 0 )
          {
            do
              v7 = *(_QWORD *)(v7 + 8);
            while ( (*(_BYTE *)(v7 + 44) & 8) != 0 );
          }
          v10 = *(_QWORD *)(v7 + 8);
          *(_QWORD *)(a1 + 3504) = v10;
          *(_QWORD *)(a1 + 3504) = sub_2EC2050(v10, v6);
        }
        sub_2EC62F0((_QWORD *)a1, v3, *(unsigned __int64 **)(a1 + 3512));
        *(_QWORD *)(a1 + 3512) = v3;
      }
    }
    (*(void (__fastcall **)(_QWORD, unsigned __int64 **, _QWORD))(**(_QWORD **)(a1 + 3472) + 120LL))(
      *(_QWORD *)(a1 + 3472),
      v2,
      i);
  }
  sub_2EC6700((_QWORD *)a1);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  if ( v12 != (__int64 *)v14 )
    _libc_free((unsigned __int64)v12);
}
