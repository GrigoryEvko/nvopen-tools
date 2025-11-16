// Function: sub_1452610
// Address: 0x1452610
//
__int64 __fastcall sub_1452610(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  int v4; // eax
  __int64 v5; // rdx
  bool v6; // sf
  __int64 result; // rax
  __int64 v8; // rax

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = v2 + 32;
      v4 = sub_16A9900(a2, v2 + 32);
      v5 = *(_QWORD *)(v2 + 24);
      if ( v4 < 0 )
        v5 = *(_QWORD *)(v2 + 16);
      if ( !v5 )
        break;
      v2 = v5;
    }
    if ( v4 >= 0 )
      goto LABEL_8;
  }
  else
  {
    v2 = a1 + 8;
  }
  result = 0;
  if ( *(_QWORD *)(a1 + 24) == v2 )
    return result;
  v8 = sub_220EF80(v2);
  v3 = v8 + 32;
  v2 = v8;
LABEL_8:
  v6 = (int)sub_16A9900(v3, a2) < 0;
  result = v2;
  if ( v6 )
    return 0;
  return result;
}
