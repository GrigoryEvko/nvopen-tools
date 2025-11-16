// Function: sub_3872670
// Address: 0x3872670
//
__int64 __fastcall sub_3872670(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 *v4; // rbx
  __int64 *v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v9; // [rsp+8h] [rbp-48h]

  v3 = sub_1456EA0(*(_QWORD *)a1, a2);
  if ( !*(_BYTE *)(a1 + 256) && (unsigned __int8)sub_1466F20(*(_QWORD *)a1, a2) )
    return 0;
  if ( !*(_WORD *)(a2 + 24) )
    return 0;
  if ( !v3 )
    return 0;
  v4 = (__int64 *)v3[4];
  v5 = (__int64 *)v3[5];
  if ( v4 == v5 )
    return 0;
  while ( 1 )
  {
    v6 = *v4;
    if ( *v4 )
    {
      if ( *(_BYTE *)(v6 + 16) > 0x17u && *(_QWORD *)v6 == sub_1456040(a2) )
      {
        v9 = sub_15F2060(v6);
        if ( v9 == sub_15F2060(a3) )
        {
          if ( sub_15CCEE0(*(_QWORD *)(*(_QWORD *)a1 + 56LL), v6, a3) )
          {
            v7 = sub_13AE450(*(_QWORD *)(*(_QWORD *)a1 + 64LL), *(_QWORD *)(v6 + 40));
            if ( !v7 || sub_1377F70(v7 + 56, *(_QWORD *)(a3 + 40)) )
              break;
          }
        }
      }
    }
    v4 += 2;
    if ( v5 == v4 )
      return 0;
  }
  return v6;
}
