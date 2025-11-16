// Function: sub_F41C30
// Address: 0xf41c30
//
__int64 __fastcall sub_F41C30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, void **a6)
{
  unsigned int v8; // r15d
  unsigned __int64 v9; // r12
  int v10; // eax
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  _QWORD v21[4]; // [rsp+20h] [rbp-60h] BYREF
  int v22; // [rsp+40h] [rbp-40h]
  char v23; // [rsp+44h] [rbp-3Ch]

  v8 = sub_D0E820(a1, a2);
  v9 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == a1 + 48 )
  {
    v11 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = v9 - 24;
    if ( (unsigned int)(v10 - 30) >= 0xB )
      v11 = 0;
  }
  v21[1] = 0;
  v21[0] = a3;
  v21[2] = a4;
  v21[3] = a5;
  v22 = 0x10000;
  v23 = 1;
  if ( (unsigned __int8)sub_D0E970(v11, v8, 0) )
  {
    v12 = sub_AA4FF0(a2);
    if ( !v12 )
      BUG();
    v13 = (unsigned int)*(unsigned __int8 *)(v12 - 24) - 39;
    if ( (unsigned int)v13 <= 0x38 && (v14 = 0x100060000000001LL, _bittest64(&v14, v13)) )
      return sub_F40FD0(a1, a2, 0, 0, v21, (__int64)a6);
    else
      return sub_F44160(v11, v8, v21, a6);
  }
  else if ( sub_AA54C0(a2) )
  {
    v16 = *(_QWORD *)(a2 + 56);
    if ( v16 )
      v16 -= 24;
    return sub_F36960(a2, (__int64 *)(v16 + 24), 0, a3, a4, a5, a6, 1);
  }
  else
  {
    v17 = sub_986580(a1);
    return sub_F36960(a1, (__int64 *)(v17 + 24), 0, a3, a4, a5, a6, 0);
  }
}
