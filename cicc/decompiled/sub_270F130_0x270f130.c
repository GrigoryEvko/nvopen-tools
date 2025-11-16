// Function: sub_270F130
// Address: 0x270f130
//
__int64 __fastcall sub_270F130(unsigned __int8 *a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int8 v3; // al
  unsigned __int8 *i; // rdi
  __int64 v6; // rdi
  unsigned __int8 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned __int8 v10; // al
  __int64 v11; // r8
  const char *v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *a1;
  if ( *a1 > 0x1Cu )
  {
    LOBYTE(v2) = v3 == 34 || v3 == 85;
    if ( (_BYTE)v2 )
      return v2;
    v2 = 1;
    if ( v3 == 60 )
      return v2;
    if ( v3 == 61 )
    {
      for ( i = (unsigned __int8 *)*((_QWORD *)a1 - 4);
            ;
            i = *(unsigned __int8 **)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)) )
      {
        v7 = sub_BD3990(i, a2);
        v6 = 23;
        v9 = (__int64)v7;
        v10 = *v7;
        if ( v10 > 0x1Cu )
        {
          if ( v10 == 85 )
          {
            v11 = *(_QWORD *)(v9 - 32);
            v6 = 21;
            if ( v11 && !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(v9 + 80) )
              v6 = (unsigned int)sub_3108960(*(_QWORD *)(v9 - 32), a2, v8);
          }
          else
          {
            v6 = 2 * (unsigned int)(v10 != 34) + 21;
          }
        }
        if ( !(unsigned __int8)sub_3108CA0(v6) )
          break;
      }
      if ( *(_BYTE *)v9 == 3 )
      {
        v2 = *(_BYTE *)(v9 + 80) & 1;
        if ( (*(_BYTE *)(v9 + 80) & 1) != 0 )
          return 1;
        v12 = sub_BD5D20(v9);
        if ( v13 > 0x15 )
        {
          v13 = *(_QWORD *)v12 ^ 0x5F636A626F5F6C01LL;
          if ( !(v13 | *((_QWORD *)v12 + 1) ^ 0x5F646E655367736DLL)
            && *((_DWORD *)v12 + 4) == 1970825574
            && *((_WORD *)v12 + 10) == 24432 )
          {
            return 1;
          }
        }
        if ( (*(_BYTE *)(v9 + 35) & 4) != 0 )
        {
          v15 = sub_B31D10(v9, a2, v13);
        }
        else
        {
          v14 = 0;
          v15 = 0;
        }
        v16[1] = v14;
        v16[0] = v15;
        if ( sub_C931B0(v16, "__message_refs", 0xEu, 0) != -1
          || sub_C931B0(v16, "__objc_classrefs", 0x10u, 0) != -1
          || sub_C931B0(v16, "__objc_superrefs", 0x10u, 0) != -1
          || sub_C931B0(v16, "__objc_methname", 0xFu, 0) != -1
          || sub_C931B0(v16, "__cstring", 9u, 0) != -1 )
        {
          return 1;
        }
        return v2;
      }
    }
  }
  else
  {
    v2 = 1;
    if ( v3 <= 0x16u )
      return v2;
  }
  return 0;
}
