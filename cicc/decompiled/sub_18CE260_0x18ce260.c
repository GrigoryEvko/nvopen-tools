// Function: sub_18CE260
// Address: 0x18ce260
//
__int64 __fastcall sub_18CE260(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int8 v2; // al
  __int64 i; // rdi
  int v5; // edi
  __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int8 v8; // al
  const char *v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13[4]; // [rsp+0h] [rbp-20h] BYREF

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 > 0x17u )
  {
    LOBYTE(v1) = v2 == 78 || v2 == 29;
    if ( (_BYTE)v1 )
      return v1;
    v1 = 1;
    if ( v2 == 53 )
      return v1;
    if ( v2 == 54 )
    {
      for ( i = *(_QWORD *)(a1 - 24); ; i = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)) )
      {
        v6 = sub_1649C60(i);
        v5 = 23;
        v7 = v6;
        v8 = *(_BYTE *)(v6 + 16);
        if ( v8 > 0x17u )
        {
          if ( v8 == 78 )
          {
            v5 = 21;
            if ( !*(_BYTE *)(*(_QWORD *)(v7 - 24) + 16LL) )
              v5 = sub_1438F00(*(_QWORD *)(v7 - 24));
          }
          else
          {
            v5 = 2 * (v8 != 29) + 21;
          }
        }
        if ( !(unsigned __int8)sub_1439C90(v5) )
          break;
      }
      if ( *(_BYTE *)(v7 + 16) == 3 )
      {
        v1 = *(_BYTE *)(v7 + 80) & 1;
        if ( (*(_BYTE *)(v7 + 80) & 1) != 0 )
          return 1;
        v9 = sub_1649960(v7);
        if ( v10 > 0x15
          && !(*(_QWORD *)v9 ^ 0x5F636A626F5F6C01LL | *((_QWORD *)v9 + 1) ^ 0x5F646E655367736DLL)
          && *((_DWORD *)v9 + 4) == 1970825574
          && *((_WORD *)v9 + 10) == 24432 )
        {
          return 1;
        }
        v11 = 0;
        v12 = 0;
        if ( (*(_BYTE *)(v7 + 34) & 0x20) != 0 )
          v12 = sub_15E61A0(v7);
        v13[1] = v11;
        v13[0] = v12;
        if ( sub_16D20C0(v13, "__message_refs", 0xEu, 0) != -1
          || sub_16D20C0(v13, "__objc_classrefs", 0x10u, 0) != -1
          || sub_16D20C0(v13, "__objc_superrefs", 0x10u, 0) != -1
          || sub_16D20C0(v13, "__objc_methname", 0xFu, 0) != -1
          || sub_16D20C0(v13, "__cstring", 9u, 0) != -1 )
        {
          return 1;
        }
        return v1;
      }
    }
  }
  else
  {
    v1 = 1;
    if ( v2 <= 0x11u )
      return v1;
  }
  return 0;
}
