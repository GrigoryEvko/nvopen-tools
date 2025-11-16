// Function: sub_157EE30
// Address: 0x157ee30
//
__int64 __fastcall sub_157EE30(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx

  v1 = sub_157ED20(a1);
  if ( !v1 )
    return a1 + 40;
  v2 = (unsigned int)*(unsigned __int8 *)(v1 + 16) - 34;
  if ( (unsigned int)v2 <= 0x36 && (v3 = 0x40018000000001LL, _bittest64(&v3, v2)) )
    return *(_QWORD *)(v1 + 32);
  else
    return v1 + 24;
}
