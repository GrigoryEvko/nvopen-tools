// Function: sub_266F4F0
// Address: 0x266f4f0
//
__int64 __fastcall sub_266F4F0(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  int v3; // edx
  unsigned __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdi

  result = sub_25282F0(*(_QWORD *)a1, a2, *(_QWORD *)(a1 + 8));
  if ( !(_BYTE)result )
  {
    v3 = *a2;
    if ( (unsigned __int8)v3 > 0x1Cu )
    {
      v4 = (unsigned int)(v3 - 34);
      if ( (unsigned __int8)v4 <= 0x33u )
      {
        v5 = 0x8000000000041LL;
        if ( _bittest64(&v5, v4) )
        {
          v6 = **(_QWORD **)(a1 + 16);
          if ( v6 )
            return (*(__int64 (__fastcall **)(__int64, unsigned __int8 *))(*(_QWORD *)v6 + 112LL))(v6, a2);
        }
      }
    }
  }
  return result;
}
