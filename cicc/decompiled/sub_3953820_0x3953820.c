// Function: sub_3953820
// Address: 0x3953820
//
bool __fastcall sub_3953820(__int64 a1, _BYTE *a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // r13
  __int64 v5; // rax

  if ( a2[16] != 17 )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 && (unsigned __int8)sub_15E0420((__int64)a2, 6) )
    return 0;
  if ( byte_50545A0 )
  {
    v2 = *(unsigned __int8 *)(*(_QWORD *)a2 + 8LL);
    if ( (unsigned __int8)v2 <= 0x10u )
    {
      v3 = 100990;
      if ( _bittest64(&v3, v2) )
      {
        if ( sub_3953740(a1 + 176, (__int64)a2) )
          return 0;
      }
    }
  }
  v4 = (unsigned __int64)dword_5054840 >> 2;
  v5 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
  return (int)v4 >= (int)sub_3952EB0((__int64)a2, v5);
}
