// Function: sub_F90AB0
// Address: 0xf90ab0
//
__int64 __fastcall sub_F90AB0(unsigned __int8 *a1, char a2)
{
  unsigned __int8 *v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rdx
  unsigned __int8 *v5; // rax

  v2 = a1;
  if ( ((a2 & 1) == 0 || !(unsigned __int8)sub_B46490((__int64)a1))
    && ((a2 & 2) == 0 || !(unsigned __int8)sub_B46420((__int64)a1) && !(unsigned __int8)sub_B46970(a1) && *a1 != 60)
    && ((a2 & 4) == 0 || sub_991A70(a1, 0, 0, 0, 0, 1u, 0)) )
  {
    if ( (unsigned __int8)(*a1 - 34) > 0x33u
      || (v3 = 0x8000000000041LL, !_bittest64(&v3, (unsigned int)*a1 - 34))
      || (unsigned int)sub_B49240((__int64)a1) != 146 )
    {
      v4 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      if ( (a1[7] & 0x40) != 0 )
      {
        v5 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        a1 = &v5[v4];
      }
      else
      {
        v5 = &a1[-v4];
      }
      if ( v5 == a1 )
        return 1;
      while ( **(_BYTE **)v5 <= 0x1Cu || *((_QWORD *)v2 + 5) != *(_QWORD *)(*(_QWORD *)v5 + 40LL) )
      {
        v5 += 32;
        if ( a1 == v5 )
          return 1;
      }
    }
  }
  return 0;
}
