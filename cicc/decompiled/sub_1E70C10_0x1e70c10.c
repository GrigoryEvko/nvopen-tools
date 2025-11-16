// Function: sub_1E70C10
// Address: 0x1e70c10
//
void __fastcall sub_1E70C10(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // r14
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

  sub_1F0A020(a1, *(_QWORD *)(a1 + 2104), 0, 0, 0, 0);
  sub_1F02930(a1 + 2128);
  sub_1E70720(a1);
  v15 = v17;
  v12 = (__int64 *)v14;
  v13 = 0x800000000LL;
  v16 = 0x800000000LL;
  sub_1E70770(a1, (__int64)&v12, (__int64)&v15);
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 2120) + 64LL))(*(_QWORD *)(a1 + 2120), a1);
  sub_1E70870(a1, v12, (unsigned int)v13, (__int64)v15, (unsigned int)v16);
  for ( i = 0; ; sub_1E70960(a1, v2, i) )
  {
    v2 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int8 *))(**(_QWORD **)(a1 + 2120) + 96LL))(
           *(_QWORD *)(a1 + 2120),
           &i);
    if ( !v2 || !(unsigned __int8)sub_1E70710(a1) )
      break;
    v3 = *(_QWORD *)(v2 + 8);
    v4 = *(_QWORD *)(a1 + 2240);
    if ( i )
    {
      if ( v3 == v4 )
      {
        v8 = *(_QWORD *)(a1 + 2248);
        if ( !v4 )
          BUG();
        if ( (*(_BYTE *)v4 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v4 + 46) & 8) != 0 )
            v4 = *(_QWORD *)(v4 + 8);
        }
        v9 = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(a1 + 2240) = v9;
        *(_QWORD *)(a1 + 2240) = sub_1E6BEE0(v9, v8);
      }
      else
      {
        sub_1E705C0(a1, *(unsigned __int64 **)(v2 + 8), (__int64 *)v4);
      }
    }
    else
    {
      v5 = sub_1E6C1C0(*(_QWORD *)(a1 + 2248), v4);
      v6 = v5;
      if ( v5 == v3 )
      {
        *(_QWORD *)(a1 + 2248) = v5;
      }
      else
      {
        v7 = *(_QWORD *)(a1 + 2240);
        if ( v3 == v7 )
        {
          if ( !v3 )
            BUG();
          if ( (*(_BYTE *)v3 & 4) == 0 && (*(_BYTE *)(v3 + 46) & 8) != 0 )
          {
            do
              v7 = *(_QWORD *)(v7 + 8);
            while ( (*(_BYTE *)(v7 + 46) & 8) != 0 );
          }
          v10 = *(_QWORD *)(v7 + 8);
          *(_QWORD *)(a1 + 2240) = v10;
          *(_QWORD *)(a1 + 2240) = sub_1E6BEE0(v10, v6);
        }
        sub_1E705C0(a1, (unsigned __int64 *)v3, *(__int64 **)(a1 + 2248));
        *(_QWORD *)(a1 + 2248) = v3;
      }
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 2120) + 112LL))(*(_QWORD *)(a1 + 2120), v2, i);
  }
  sub_1E709A0(a1);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
  if ( v12 != (__int64 *)v14 )
    _libc_free((unsigned __int64)v12);
}
