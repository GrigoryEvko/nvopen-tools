// Function: sub_22BEBD0
// Address: 0x22bebd0
//
__int64 __fastcall sub_22BEBD0(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  __int64 v4; // rdi
  unsigned __int8 *v5; // rdx
  unsigned __int8 v6; // dl
  char v7; // dl
  __int64 *v8; // rbx
  __int64 v9; // [rsp+10h] [rbp-20h]
  unsigned __int8 v10; // [rsp+18h] [rbp-18h]

  result = *a2;
  if ( (unsigned __int8)result > 0x1Cu )
  {
    result = (unsigned int)(result - 29);
  }
  else
  {
    if ( (_BYTE)result != 5 )
      return result;
    result = *((unsigned __int16 *)a2 + 1);
  }
  if ( (_DWORD)result == 47 )
  {
    v4 = *(_QWORD *)a1;
    v5 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v9 = sub_9208B0(v4, *(_QWORD *)(*(_QWORD *)v5 + 8LL));
    v10 = v6;
    result = sub_9208B0(*(_QWORD *)a1, *((_QWORD *)a2 + 1));
    if ( result == v9 )
    {
      result = v10;
      if ( v7 == v10 )
      {
        if ( (a2[7] & 0x40) != 0 )
          v8 = (__int64 *)*((_QWORD *)a2 - 1);
        else
          v8 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        result = *v8;
        if ( *v8 )
          **(_QWORD **)(a1 + 8) = result;
      }
    }
  }
  return result;
}
