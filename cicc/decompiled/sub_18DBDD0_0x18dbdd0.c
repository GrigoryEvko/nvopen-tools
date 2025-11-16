// Function: sub_18DBDD0
// Address: 0x18dbdd0
//
void __fastcall sub_18DBDD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  char v9; // al
  __int64 *v10; // rax
  __int64 v11; // r12
  unsigned __int8 v12; // al
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // [rsp+0h] [rbp-50h] BYREF
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v17[8]; // [rsp+10h] [rbp-40h] BYREF

  v17[1] = &v15;
  v17[2] = &v16;
  v9 = *(_BYTE *)(a1 + 2);
  v16 = a2;
  v15 = a3;
  v17[0] = a1;
  if ( v9 == 4 )
  {
    if ( (unsigned __int8)sub_18DC540(v15, a4, a5, a6) )
      sub_18DB890(a1, 3);
  }
  else if ( (unsigned __int8)(v9 - 5) <= 1u )
  {
    if ( (unsigned __int8)sub_18DC540(v15, a4, a5, a6) )
    {
      sub_18DB8A0((__int64)v17, 3);
      return;
    }
    if ( *(_BYTE *)(a1 + 2) == 5 && (unsigned __int8)sub_1439C40(a6) )
      goto LABEL_17;
    if ( a6 == 1 )
    {
      v10 = (*(_BYTE *)(v15 + 23) & 0x40) != 0
          ? *(__int64 **)(v15 - 8)
          : (__int64 *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
      v11 = sub_1649C60(*v10);
      v12 = *(_BYTE *)(v11 + 16);
      if ( v12 > 0x17u )
      {
        if ( v12 == 78 )
        {
          v14 = *(_QWORD *)(v11 - 24);
          v13 = 21;
          if ( !*(_BYTE *)(v14 + 16) )
            v13 = (unsigned int)sub_1438F00(v14);
        }
        else
        {
          if ( v12 != 29 )
            return;
          v13 = 21;
        }
        if ( (unsigned __int8)sub_18DC540(v11, a4, a5, v13) )
LABEL_17:
          sub_18DB8A0((__int64)v17, 4);
      }
    }
  }
}
