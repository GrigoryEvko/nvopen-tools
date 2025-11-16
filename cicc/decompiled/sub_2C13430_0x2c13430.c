// Function: sub_2C13430
// Address: 0x2c13430
//
__int64 __fastcall sub_2C13430(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // r15
  unsigned __int64 v5; // rax
  int v6; // edx
  _QWORD *v7; // r12
  unsigned __int64 v8; // rax
  __int64 v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  _QWORD *v14; // [rsp+8h] [rbp-78h]
  __int64 v15[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v16; // [rsp+40h] [rbp-40h]

  v15[0] = *(_QWORD *)(a1 + 88);
  if ( v15[0] )
    sub_2AAAFA0(v15);
  sub_2BF1A90(a2, (__int64)v15);
  sub_9C6650(v15);
  v2 = sub_2BFB120(a2, **(_QWORD **)(a1 + 48), (unsigned int *)(a2 + 16));
  v3 = *(_QWORD *)(a2 + 104);
  v4 = v2;
  v5 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == v3 + 48 )
  {
    v14 = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    v6 = *(unsigned __int8 *)(v5 - 24);
    v7 = 0;
    v8 = v5 - 24;
    if ( (unsigned int)(v6 - 30) < 0xB )
      v7 = (_QWORD *)v8;
    v14 = v7;
  }
  v9 = *(_QWORD *)(a2 + 904);
  v16 = 257;
  v10 = sub_BD2C40(72, 3u);
  v11 = (__int64)v10;
  if ( v10 )
    sub_B4C9A0((__int64)v10, v3, 0, v4, 3u, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
    *(_QWORD *)(v9 + 88),
    v11,
    v15,
    *(_QWORD *)(v9 + 56),
    *(_QWORD *)(v9 + 64));
  sub_94AAF0((unsigned int **)v9, v11);
  if ( *(_QWORD *)(v11 - 32) )
  {
    v12 = *(_QWORD *)(v11 - 24);
    **(_QWORD **)(v11 - 16) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(v11 - 16);
  }
  *(_QWORD *)(v11 - 32) = 0;
  return sub_B43D60(v14);
}
