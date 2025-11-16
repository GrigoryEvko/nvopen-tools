// Function: sub_199B0A0
// Address: 0x199b0a0
//
__int64 __fastcall sub_199B0A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v5; // ax
  unsigned int v6; // r15d
  __int16 v7; // ax
  _QWORD *v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // r14
  char v17[32]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int8 v18; // [rsp+20h] [rbp-40h]

  while ( 1 )
  {
    v5 = *(_WORD *)(a1 + 24);
    if ( v5 <= 3u )
    {
      while ( v5 )
      {
        a1 = *(_QWORD *)(a1 + 32);
        v5 = *(_WORD *)(a1 + 24);
        if ( v5 > 3u )
          goto LABEL_4;
      }
      return 0;
    }
LABEL_4:
    if ( v5 == 10 )
      return 0;
    sub_199AF80((__int64)v17, a2, a1);
    v6 = v18;
    if ( !v18 )
      return 0;
    v7 = *(_WORD *)(a1 + 24);
    if ( v7 == 4 )
    {
      v8 = *(_QWORD **)(a1 + 32);
      v9 = &v8[*(_QWORD *)(a1 + 40)];
      if ( v9 != v8 )
      {
        v10 = *(_QWORD **)(a1 + 32);
        while ( !(unsigned __int8)sub_199B0A0(*v10, a2, a3) )
        {
          if ( v9 == ++v10 )
            return 0;
        }
        return v6;
      }
      return 0;
    }
    if ( v7 != 5 )
    {
LABEL_18:
      if ( v7 == 7 )
        return (unsigned int)sub_1993E40(a1, a3) ^ 1;
      return v6;
    }
    if ( *(_QWORD *)(a1 + 40) != 2 )
      return v6;
    v12 = *(_QWORD **)(a1 + 32);
    v13 = v12[1];
    if ( *(_WORD *)(*v12 + 24LL) )
      break;
    a1 = v12[1];
  }
  if ( *(_WORD *)(v13 + 24) == 10 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v13 - 8) + 8LL);
    if ( v14 )
    {
      while ( 1 )
      {
        v15 = sub_1648700(v14);
        v16 = (__int64)v15;
        if ( *((_BYTE *)v15 + 16) == 39 && sub_1456C80(a3, *v15) )
          break;
        v14 = *(_QWORD *)(v14 + 8);
        if ( !v14 )
        {
          v7 = *(_WORD *)(a1 + 24);
          goto LABEL_18;
        }
      }
      LOBYTE(v6) = a1 == sub_146F1B0(a3, v16);
    }
  }
  return v6;
}
