// Function: sub_155F320
// Address: 0x155f320
//
__int64 __fastcall sub_155F320(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v3 = *a1;
  if ( !v3 )
    return 0;
  v2 = *(_QWORD *)(v3 + 8);
  if ( _bittest64(&v2, a2) )
    return sub_155F2B0(v3, a2);
  else
    return 0;
}
