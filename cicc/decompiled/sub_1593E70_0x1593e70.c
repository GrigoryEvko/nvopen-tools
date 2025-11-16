// Function: sub_1593E70
// Address: 0x1593e70
//
__int64 __fastcall sub_1593E70(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return 0;
  while ( 1 )
  {
    v2 = sub_1648700(v1);
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 16) - 4) > 0xCu )
      break;
    result = sub_1593E70(v2);
    if ( !(_BYTE)result )
    {
      v1 = *(_QWORD *)(v1 + 8);
      if ( v1 )
        continue;
    }
    return result;
  }
  return 1;
}
