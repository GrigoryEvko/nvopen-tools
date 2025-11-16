// Function: sub_26F6120
// Address: 0x26f6120
//
__int64 __fastcall sub_26F6120(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // r13

  v1 = a1 + 360;
  result = sub_ACD6D0(**(__int64 ***)a1);
  v3 = *(_QWORD *)(a1 + 376);
  if ( v3 != a1 + 360 )
  {
    v4 = result;
    do
    {
      while ( *(_DWORD *)(v3 + 40) )
      {
        result = sub_220EEE0(v3);
        v3 = result;
        if ( v1 == result )
          return result;
      }
      sub_BD84D0(*(_QWORD *)(v3 + 32), v4);
      sub_B43D60(*(_QWORD **)(v3 + 32));
      result = sub_220EEE0(v3);
      v3 = result;
    }
    while ( v1 != result );
  }
  return result;
}
