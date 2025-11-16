// Function: sub_29AA4F0
// Address: 0x29aa4f0
//
bool __fastcall sub_29AA4F0(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rbx
  __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  __int64 v10; // rax
  __int64 v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = (_QWORD *)(a1 + 48);
  v3 = *(_QWORD **)(a1 + 56);
  if ( v3 == (_QWORD *)(a1 + 48) )
    return 0;
  while ( 1 )
  {
    if ( !v3 )
      BUG();
    v5 = v3[3];
    if ( v5 )
    {
      if ( *((_BYTE *)v3 - 24) != 85 )
        break;
      v10 = *(v3 - 7);
      if ( !v10
        || *(_BYTE *)v10
        || *(_QWORD *)(v10 + 24) != v3[7]
        || (*(_BYTE *)(v10 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v10 + 36) - 68) > 3 )
      {
        break;
      }
    }
    v3 = (_QWORD *)v3[1];
    if ( v2 == v3 )
      return 0;
  }
  v6 = *a2;
  v11[0] = v3[3];
  sub_B96E90((__int64)v11, v5, 1);
  if ( (__int64 *)(v6 + 48) == v11 )
  {
    if ( v11[0] )
      sub_B91220(v6 + 48, v11[0]);
  }
  else
  {
    v7 = *(_QWORD *)(v6 + 48);
    if ( v7 )
      sub_B91220(v6 + 48, v7);
    v8 = (unsigned __int8 *)v11[0];
    *(_QWORD *)(v6 + 48) = v11[0];
    if ( v8 )
      sub_B976B0((__int64)v11, v8, v6 + 48);
  }
  return v2 != v3;
}
