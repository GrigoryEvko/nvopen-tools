// Function: sub_AC2F40
// Address: 0xac2f40
//
__int64 __fastcall sub_AC2F40(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  __int64 v5; // rdx
  __int64 v7; // rax
  unsigned __int8 *v8; // r12
  unsigned __int8 *v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx

  v5 = *a1;
  if ( (unsigned __int8)v5 > 0xBu )
  {
    LOBYTE(v4) = (unsigned int)v5 <= 0x15;
    return v4;
  }
  LOBYTE(v4) = (_BYTE)v5 == 5 || (unsigned int)v5 > 8;
  if ( !(_BYTE)v4 )
    return v4;
  v7 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  if ( (a1[7] & 0x40) != 0 )
  {
    v9 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
    v8 = &v9[v7];
  }
  else
  {
    v8 = a1;
    v9 = &a1[-v7];
  }
  if ( v8 == v9 )
    return v4;
  if ( (unsigned __int8)sub_AC2F40(*(_QWORD *)v9, a2, v5, a4) )
  {
    while ( 1 )
    {
      v9 += 32;
      if ( v8 == v9 )
        break;
      if ( !(unsigned __int8)sub_AC2F40(*(_QWORD *)v9, a2, v10, v11) )
        return 0;
    }
    return v4;
  }
  return 0;
}
