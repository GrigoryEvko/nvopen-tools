// Function: sub_FFE160
// Address: 0xffe160
//
__int64 __fastcall sub_FFE160(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v5; // r14
  __int64 v6; // r15

  v3 = qword_4F8EB68;
  if ( (_DWORD)qword_4F8EB68 )
  {
    v5 = a3;
    v3 = 0;
    do
    {
      if ( *(_DWORD *)(a1 + 8) <= v3 )
        break;
      v6 = *(_QWORD *)(*(_QWORD *)a1 + 16LL * v3 + 8);
      if ( !(unsigned __int8)sub_FFE120(a1, v6, a3, v5) )
        break;
      v5 -= v6;
      ++v3;
    }
    while ( (unsigned int)qword_4F8EB68 > v3 );
  }
  return v3;
}
