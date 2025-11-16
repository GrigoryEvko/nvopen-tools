// Function: sub_8E0610
// Address: 0x8e0610
//
__int64 __fastcall sub_8E0610(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned int v6; // r13d
  __int64 v8; // rbx
  __int64 v9; // rax
  char i; // dl
  char v11; // cl
  _QWORD *v12; // rax
  _BYTE v13[64]; // [rsp+0h] [rbp-40h] BYREF

  v2 = a2;
  v3 = a1;
  if ( (unsigned int)sub_8D2FB0(a1) )
    v3 = sub_8D46C0(a1);
  while ( *(_BYTE *)(v3 + 140) == 12 )
    v3 = *(_QWORD *)(v3 + 160);
  if ( (unsigned int)sub_8D2FB0(a2) )
    v2 = sub_8D46C0(a2);
  while ( *(_BYTE *)(v2 + 140) == 12 )
    v2 = *(_QWORD *)(v2 + 160);
  if ( v2 == v3 )
    return 1;
  v6 = sub_8D97D0(v3, v2, 0, v4, v5);
  if ( v6 )
  {
    return 1;
  }
  else
  {
    if ( (unsigned int)sub_8D2F30(v3, v2) )
    {
      v8 = sub_8D46C0(v3);
      v9 = sub_8D46C0(v2);
      for ( i = *(_BYTE *)(v8 + 140); i == 12; i = *(_BYTE *)(v8 + 140) )
        v8 = *(_QWORD *)(v8 + 160);
      while ( 1 )
      {
        v11 = *(_BYTE *)(v9 + 140);
        if ( v11 != 12 )
          break;
        v9 = *(_QWORD *)(v9 + 160);
      }
      if ( (unsigned __int8)(i - 9) <= 2u && (unsigned __int8)(v11 - 9) <= 2u )
      {
        v2 = v9;
        v3 = v8;
      }
    }
    if ( sub_8D3A70(v3) && sub_8D3A70(v2) )
    {
      v12 = sub_8D5CE0(v2, v3);
      if ( v12 && (v12[12] & 4) == 0 )
        return (unsigned int)sub_87DF20((__int64)v12) != 0;
    }
    else if ( (unsigned int)sub_8D2E30(v3) && (unsigned int)sub_8D2E30(v2) )
    {
      return (unsigned int)sub_8DFA20(v2, 0, 0, 0, 0, v3, 0, 1, 0, (__int64)v13, 0) != 0;
    }
  }
  return v6;
}
