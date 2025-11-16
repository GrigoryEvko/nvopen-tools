// Function: sub_1843420
// Address: 0x1843420
//
__int64 __fastcall sub_1843420(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax

  v3 = *(_QWORD *)(a2 + 8);
  if ( !v3 )
    return 1;
  do
  {
    result = sub_18430C0(a1, v3, a3, 0xFFFFFFFF);
    if ( !(_DWORD)result )
      break;
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v3 );
  return result;
}
