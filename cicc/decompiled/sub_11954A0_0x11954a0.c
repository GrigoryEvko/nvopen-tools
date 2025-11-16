// Function: sub_11954A0
// Address: 0x11954a0
//
bool __fastcall sub_11954A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __int64 v5; // r12
  int v7; // r13d
  unsigned int v8; // eax
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned int v11; // r12d
  bool v12; // [rsp+Fh] [rbp-31h]
  unsigned __int64 *v13; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-28h]

  result = 0;
  v5 = *(_QWORD *)(a2 + 8);
  if ( v5 == *(_QWORD *)(a4 + 8) )
  {
    v7 = sub_BCB060(*(_QWORD *)(a1 + 8));
    LODWORD(a3) = sub_BCB060(*(_QWORD *)(a3 + 8));
    v8 = sub_BCB060(v5);
    v9 = (unsigned int)(v7 + a3 - 2);
    v14 = v8;
    if ( v8 > 0x40 )
    {
      sub_C43690((__int64)&v13, -1, 1);
      v11 = v14;
      if ( v14 > 0x40 )
      {
        if ( v11 - (unsigned int)sub_C444A0((__int64)&v13) <= 0x40 )
        {
          result = *v13 >= v9;
        }
        else
        {
          result = 1;
          if ( !v13 )
            return result;
        }
        v12 = result;
        j_j___libc_free_0_0(v13);
        return v12;
      }
    }
    else
    {
      v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
      if ( !v8 )
        v10 = 0;
      v13 = (unsigned __int64 *)v10;
    }
    return (unsigned __int64)v13 >= v9;
  }
  return result;
}
