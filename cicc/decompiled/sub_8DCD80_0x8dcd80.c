// Function: sub_8DCD80
// Address: 0x8dcd80
//
void __fastcall sub_8DCD80(__int64 a1)
{
  char v2; // al
  __int64 v3; // rax
  __int64 ***v4; // rbx
  _BOOL4 v5; // eax
  __int64 **i; // r12
  __int64 v7; // rax
  _DWORD v8[5]; // [rsp+Ch] [rbp-14h] BYREF

  while ( 1 )
  {
    while ( 1 )
    {
      v2 = *(_BYTE *)(a1 + 140);
      if ( v2 != 12 )
        break;
      a1 = *(_QWORD *)(a1 + 160);
    }
    if ( (unsigned __int8)(v2 - 9) <= 2u )
      break;
    if ( sub_8D32B0(a1) )
    {
      a1 = sub_8D46C0(a1);
    }
    else
    {
      if ( *(_BYTE *)(a1 + 140) != 13 )
        return;
      v3 = sub_8D4890(a1);
      sub_8DCD80(v3);
      a1 = sub_8D4870(a1);
    }
  }
  v4 = *(__int64 ****)(a1 + 168);
  if ( (*(_DWORD *)(a1 + 176) & 0x11000) == 0x1000
    || (v7 = sub_735B60(a1, 0)) == 0
    || (*(_BYTE *)(v7 + 193) & 0x20) != 0
    || *(_DWORD *)(v7 + 160)
    || *(_QWORD *)(v7 + 344) )
  {
    v5 = sub_7E3E60(a1, v8);
    if ( v8[0] | v5 )
      sub_71BC30(a1);
    for ( i = *v4; i; i = (__int64 **)*i )
    {
      if ( ((_BYTE)i[12] & 1) != 0 )
        sub_8DCD80(i[5]);
    }
  }
}
