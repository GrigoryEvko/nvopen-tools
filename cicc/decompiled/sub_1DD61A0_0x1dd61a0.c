// Function: sub_1DD61A0
// Address: 0x1dd61a0
//
__int64 __fastcall sub_1DD61A0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  unsigned int v4; // r8d

  v1 = *(_QWORD *)(a1 + 88);
  v2 = *(_QWORD *)(a1 + 96);
  if ( v1 == v2 )
  {
    return 0;
  }
  else
  {
    do
    {
      v4 = *(unsigned __int8 *)(*(_QWORD *)v1 + 180LL);
      if ( (_BYTE)v4 )
        break;
      v1 += 8;
    }
    while ( v2 != v1 );
  }
  return v4;
}
