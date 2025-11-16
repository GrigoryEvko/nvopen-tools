// Function: sub_271D970
// Address: 0x271d970
//
void __fastcall sub_271D970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  unsigned __int8 v9; // al
  unsigned __int8 **v10; // rax
  __int64 v11; // rdx
  unsigned __int8 *v12; // r12
  unsigned __int8 v13; // al
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // [rsp+0h] [rbp-50h] BYREF
  __int64 v17; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v18[8]; // [rsp+10h] [rbp-40h] BYREF

  v18[1] = &v16;
  v18[2] = &v17;
  v9 = *(_BYTE *)(a1 + 2);
  v17 = a2;
  v16 = a3;
  v18[0] = a1;
  if ( v9 == 5 )
  {
    if ( (unsigned __int8)sub_3181380(v16, a4, a5, a6) )
    {
      sub_271D2F0((__int64)v18, 3);
      return;
    }
    if ( a6 == 1 )
    {
      v10 = (*(_BYTE *)(v16 + 7) & 0x40) != 0
          ? *(unsigned __int8 ***)(v16 - 8)
          : (unsigned __int8 **)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF));
      v12 = sub_BD3990(*v10, a4);
      v13 = *v12;
      if ( *v12 > 0x1Cu )
      {
        if ( v13 == 85 )
        {
          v15 = *((_QWORD *)v12 - 4);
          v14 = 21;
          if ( v15 && !*(_BYTE *)v15 && *(_QWORD *)(v15 + 24) == *((_QWORD *)v12 + 10) )
            v14 = (unsigned int)sub_3108960(v15, a4, v11);
        }
        else
        {
          if ( v13 != 34 )
            return;
          v14 = 21;
        }
        if ( (unsigned __int8)sub_3181380(v12, a4, a5, v14) )
          sub_271D2F0((__int64)v18, 4);
      }
    }
  }
  else if ( v9 <= 5u )
  {
    if ( v9 == 1 )
      BUG();
    if ( v9 == 4 )
    {
      if ( (unsigned __int8)sub_3181380(v16, a4, a5, a6) )
        sub_271D2E0(a1, 3);
    }
  }
}
