// Function: sub_1E77680
// Address: 0x1e77680
//
void __fastcall sub_1E77680(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rax
  unsigned int v8; // [rsp+Ch] [rbp-E4h]
  unsigned __int8 i; // [rsp+1Fh] [rbp-D1h] BYREF
  __int64 *v10; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v11; // [rsp+28h] [rbp-C8h]
  _BYTE v12[64]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE *v13; // [rsp+70h] [rbp-80h] BYREF
  __int64 v14; // [rsp+78h] [rbp-78h]
  _BYTE v15[112]; // [rsp+80h] [rbp-70h] BYREF

  sub_1E77580(a1);
  sub_1F02930(a1 + 2128);
  sub_1E70720(a1);
  v13 = v15;
  v10 = (__int64 *)v12;
  v11 = 0x800000000LL;
  v14 = 0x800000000LL;
  sub_1E70770(a1, (__int64)&v10, (__int64)&v13);
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 2120) + 64LL))(*(_QWORD *)(a1 + 2120), a1);
  sub_1E71D10(a1, v10, (unsigned int)v11, (__int64)v13, (unsigned int)v14);
  for ( i = 0; ; sub_1E70960(a1, v1, i) )
  {
    v1 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int8 *))(**(_QWORD **)(a1 + 2120) + 96LL))(
           *(_QWORD *)(a1 + 2120),
           &i);
    if ( !v1 || !(unsigned __int8)sub_1E70710(a1) )
      break;
    sub_1E71D40(a1, v1, i);
    v2 = *(_QWORD *)(a1 + 2280);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 8);
      if ( *(_QWORD *)(v2 + 16) == v3 )
      {
        v6 = 0;
        v5 = 1;
        v4 = 0;
      }
      else
      {
        v4 = *(_DWORD *)(v3 + 8LL * *(unsigned int *)(v1 + 192) + 4);
        v5 = 1LL << v4;
        v6 = 8LL * (v4 >> 6);
      }
      v7 = (_QWORD *)(*(_QWORD *)(a1 + 2288) + v6);
      if ( (v5 & *v7) == 0 )
      {
        v8 = v4;
        *v7 |= v5;
        sub_1F05610(*(_QWORD *)(a1 + 2280), v4);
        (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 2120) + 104LL))(*(_QWORD *)(a1 + 2120), v8);
      }
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 2120) + 112LL))(*(_QWORD *)(a1 + 2120), v1, i);
  }
  sub_1E709A0(a1);
  if ( v13 != v15 )
    _libc_free((unsigned __int64)v13);
  if ( v10 != (__int64 *)v12 )
    _libc_free((unsigned __int64)v10);
}
