// Function: sub_1E163A0
// Address: 0x1e163a0
//
__int64 __fastcall sub_1E163A0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  int i; // ecx
  char v4; // dl
  _BYTE *v5; // rdx

  v1 = *(_QWORD *)(a1 + 16);
  result = *(unsigned __int8 *)(v1 + 4);
  if ( (*(_BYTE *)(v1 + 8) & 1) != 0 )
  {
    for ( i = *(_DWORD *)(a1 + 40); i != (_DWORD)result; result = (unsigned int)(result + 1) )
    {
      v5 = (_BYTE *)(*(_QWORD *)(a1 + 32) + 40LL * (unsigned int)result);
      if ( *v5 )
        break;
      v4 = v5[3];
      if ( (v4 & 0x10) == 0 )
        break;
      if ( (v4 & 0x20) != 0 )
        break;
    }
  }
  return result;
}
