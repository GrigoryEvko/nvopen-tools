// Function: sub_25B02B0
// Address: 0x25b02b0
//
__int64 __fastcall sub_25B02B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 result; // rax

  v3 = *(_QWORD *)(a2 + 16);
  if ( !v3 )
    return 1;
  do
  {
    result = sub_25AFFC0(a1, v3, a3, 0xFFFFFFFF);
    if ( !(_DWORD)result )
      break;
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v3 );
  return result;
}
