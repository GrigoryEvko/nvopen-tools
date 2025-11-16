// Function: sub_21F29B0
// Address: 0x21f29b0
//
__int64 __fastcall sub_21F29B0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  unsigned int v6; // r15d
  __int64 v8; // rbx
  __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 16) == 17 )
  {
    if ( byte_4FD42E8[0] || (unsigned __int8)sub_15E04B0(a2) )
    {
      *a3 = a2;
      return 1;
    }
    *a3 = a2;
  }
  v8 = sub_146F1B0(a4, a2);
  if ( *(_WORD *)(v8 + 24) == 7 )
  {
    do
    {
      v8 = **(_QWORD **)(v8 + 32);
      v9 = *(_WORD *)(v8 + 24);
      if ( v9 == 10 )
      {
        v10 = *(_QWORD *)(v8 - 8);
        if ( *(_BYTE *)(v10 + 16) != 17 )
          break;
        v6 = byte_4FD42E8[0];
        if ( byte_4FD42E8[0] )
        {
          *a3 = v10;
          return v6;
        }
        v15 = *(_QWORD *)(v8 - 8);
        v6 = sub_15E04B0(v10);
        *a3 = v15;
        if ( (_BYTE)v6 )
          return v6;
        v9 = *(_WORD *)(v8 + 24);
      }
      if ( v9 == 4 )
      {
        if ( (unsigned __int8)sub_21F2180(v8, a3) )
          return 1;
        v9 = *(_WORD *)(v8 + 24);
      }
    }
    while ( v9 == 7 );
  }
  v11 = sub_146F1B0(a4, a2);
  if ( *(_WORD *)(v11 + 24) == 4 && (unsigned __int8)sub_21F2180(v11, a3) )
  {
    return 1;
  }
  else
  {
    LOBYTE(v12) = sub_1CCB410(a2);
    v6 = v12;
    if ( (_BYTE)v12
      && (v13 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a4 + 24) + 40LL)),
          v14 = sub_14AD280(a2, v13, 6u),
          v14 == sub_14AD280(v14, v13, 6u)) )
    {
      *a3 = v14;
    }
    else
    {
      return 0;
    }
  }
  return v6;
}
