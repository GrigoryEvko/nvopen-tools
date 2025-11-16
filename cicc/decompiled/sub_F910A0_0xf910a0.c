// Function: sub_F910A0
// Address: 0xf910a0
//
__int64 __fastcall sub_F910A0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // r8
  __int64 v4; // r12
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( !v1 )
    return 0;
  while ( 1 )
  {
    v2 = *(_QWORD *)(v1 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v2 - 30) <= 0xAu )
      break;
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return 0;
  }
  v3 = 0;
LABEL_5:
  v4 = v3;
  result = sub_AA54C0(*(_QWORD *)(v2 + 40));
  v3 = result;
  if ( !result || v4 && result != v4 )
    return 0;
  while ( 1 )
  {
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return result;
    v2 = *(_QWORD *)(v1 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v2 - 30) <= 0xAu )
      goto LABEL_5;
  }
}
