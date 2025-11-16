// Function: sub_2919680
// Address: 0x2919680
//
__int64 __fastcall sub_2919680(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 *v5; // r15
  unsigned __int64 v6; // r14
  __int64 i; // r8
  __int64 **v9; // r15
  __int64 **v10; // rbx
  __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+8h] [rbp-48h]

  v3 = sub_9208B0(a3, *(_QWORD *)(a2 + 24));
  if ( (v3 & 7) != 0 )
    return 0;
  v4 = *(__int64 **)(a1 + 16);
  v5 = *(__int64 **)(a1 + 24);
  v6 = v3 >> 3;
  for ( i = a3; v5 != v4; i = v12 )
  {
    v12 = i;
    if ( !sub_2919400((__int64 *)a1, v4, a2, v6, i) )
      return 0;
    v4 += 3;
  }
  v9 = *(__int64 ***)(a1 + 32);
  v10 = &v9[*(unsigned int *)(a1 + 40)];
  if ( v10 != v9 )
  {
    while ( 1 )
    {
      v13 = i;
      if ( !sub_2919400((__int64 *)a1, *v9, a2, v6, i) )
        break;
      ++v9;
      i = v13;
      if ( v10 == v9 )
        return 1;
    }
    return 0;
  }
  return 1;
}
