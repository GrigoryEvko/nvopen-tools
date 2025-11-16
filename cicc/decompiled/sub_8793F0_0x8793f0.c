// Function: sub_8793F0
// Address: 0x8793f0
//
__int64 *__fastcall sub_8793F0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v3; // r12
  __int64 v5; // rax
  __int64 v6; // rax

  v3 = **(__int64 ***)(a2 + 168);
  if ( !v3 )
    return 0;
  while ( 1 )
  {
    v6 = v3[5];
    if ( v6 == a1 || a1 && v6 && dword_4F07588 && (v5 = *(_QWORD *)(v6 + 32), *(_QWORD *)(a1 + 32) == v5) && v5 )
    {
      if ( (v3[12] & 4) == 0 || a3 == v3 || (unsigned int)sub_8D5D50(v3, a3) )
        break;
    }
    v3 = (__int64 *)*v3;
    if ( !v3 )
      return 0;
  }
  return v3;
}
