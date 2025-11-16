// Function: sub_1584660
// Address: 0x1584660
//
__int64 __fastcall sub_1584660(_QWORD *a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __int64 v4; // r13
  unsigned int v5; // r14d
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // [rsp+8h] [rbp-D8h]
  __int64 v18; // [rsp+8h] [rbp-D8h]
  __int64 v19; // [rsp+18h] [rbp-C8h]
  _BYTE *v20; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+28h] [rbp-B8h]
  _BYTE v22[176]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = *(_BYTE *)(a3 + 16);
  if ( v3 == 9 )
    return sub_1599EF0(*a1);
  if ( v3 != 13 )
    return 0;
  v4 = *a1;
  v5 = *(_DWORD *)(a3 + 32);
  v17 = *(_QWORD *)(*a1 + 32LL);
  v7 = (unsigned int)v17;
  if ( v5 > 0x40 )
  {
    if ( v5 - (unsigned int)sub_16A57B0(a3 + 24) <= 0x40 && (unsigned __int64)(unsigned int)v17 > **(_QWORD **)(a3 + 24) )
      goto LABEL_5;
    return sub_1599EF0(v4);
  }
  if ( (unsigned __int64)(unsigned int)v17 <= *(_QWORD *)(a3 + 24) )
    return sub_1599EF0(v4);
LABEL_5:
  v20 = v22;
  v21 = 0x1000000000LL;
  if ( (unsigned int)v17 > 0x10uLL )
    sub_16CD150(&v20, v22, (unsigned int)v17, 8);
  v8 = sub_16498A0(a1);
  v9 = sub_1643350(v8);
  if ( *(_DWORD *)(a3 + 32) <= 0x40u )
    v10 = *(_QWORD *)(a3 + 24);
  else
    v10 = **(_QWORD **)(a3 + 24);
  v11 = 0;
  if ( (_DWORD)v17 )
  {
    do
    {
      while ( v10 == v11 )
      {
        v15 = (unsigned int)v21;
        if ( (unsigned int)v21 >= HIDWORD(v21) )
        {
          sub_16CD150(&v20, v22, 0, 8);
          v15 = (unsigned int)v21;
        }
        ++v11;
        *(_QWORD *)&v20[8 * v15] = a2;
        LODWORD(v21) = v21 + 1;
        if ( v7 == v11 )
          goto LABEL_18;
      }
      v12 = sub_159C470(v9, v11, 0);
      v13 = sub_15A37D0(a1, v12, 0);
      v14 = (unsigned int)v21;
      if ( (unsigned int)v21 >= HIDWORD(v21) )
      {
        v18 = v13;
        sub_16CD150(&v20, v22, 0, 8);
        v14 = (unsigned int)v21;
        v13 = v18;
      }
      ++v11;
      *(_QWORD *)&v20[8 * v14] = v13;
      LODWORD(v21) = v21 + 1;
    }
    while ( v7 != v11 );
  }
LABEL_18:
  result = sub_15A01B0(v20, (unsigned int)v21);
  if ( v20 != v22 )
  {
    v19 = result;
    _libc_free((unsigned __int64)v20);
    return v19;
  }
  return result;
}
