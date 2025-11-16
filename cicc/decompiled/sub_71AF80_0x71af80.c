// Function: sub_71AF80
// Address: 0x71af80
//
__int64 __fastcall sub_71AF80(__int64 a1)
{
  __int64 result; // rax
  __int64 ***v2; // r13
  __int64 i; // r12
  __int64 **j; // rbx

  result = *(_QWORD *)(a1 + 176) & 0x100000100LL;
  if ( result == 256 )
  {
    *(_BYTE *)(a1 + 180) |= 1u;
    v2 = *(__int64 ****)(a1 + 168);
    result = dword_4F068EC;
    if ( dword_4F068EC || (result = *(_DWORD *)(a1 + 176) & 0x11000, (_DWORD)result == 4096) )
    {
      if ( (*(_BYTE *)(a1 + 176) & 0x40) != 0 )
      {
        result = (__int64)v2[19];
        for ( i = *(_QWORD *)(result + 144); i; i = *(_QWORD *)(i + 112) )
        {
          result = *(_BYTE *)(i + 192) & 0xA;
          if ( (_BYTE)result == 2 )
          {
            sub_8AD0D0(*(_QWORD *)i, 1, 1);
            result = sub_75BF90(a1);
          }
        }
      }
    }
    for ( j = *v2; j; j = (__int64 **)*j )
    {
      if ( ((_BYTE)j[12] & 1) != 0 )
        result = sub_71AF80(j[5]);
    }
  }
  return result;
}
