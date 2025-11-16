// Function: sub_88DA60
// Address: 0x88da60
//
__int64 __fastcall sub_88DA60(__int64 ***a1, unsigned int a2)
{
  __int64 **v2; // rbx
  __int64 result; // rax
  bool v4; // zf
  __int64 v5; // rdx

  v2 = *a1;
  if ( *a1 )
  {
    sub_8603B0((__int64)a1, *((_DWORD *)v2[1] + 10), a2, 0);
    result = 0;
    do
    {
      v4 = ((_BYTE)v2[7] & 0x10) == 0;
      v2 = (__int64 **)*v2;
      if ( !v4 )
        result = 1;
    }
    while ( v2 );
    v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (_DWORD)result )
      *(_BYTE *)(v5 + 7) |= 1u;
    *(_BYTE *)(v5 + 10) |= 4u;
  }
  else
  {
    sub_8603B0((__int64)a1, -1, a2, 0);
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 10) |= 4u;
    return (__int64)qword_4F04C68;
  }
  return result;
}
