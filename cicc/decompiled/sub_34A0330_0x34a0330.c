// Function: sub_34A0330
// Address: 0x34a0330
//
__int64 __fastcall sub_34A0330(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  char v4; // dl
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // rax
  bool v8; // zf

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = v2 + 32;
      v4 = sub_34A0190(a2, v2 + 32);
      v5 = *(_QWORD *)(v2 + 24);
      if ( v4 )
        v5 = *(_QWORD *)(v2 + 16);
      if ( !v5 )
        break;
      v2 = v5;
    }
    if ( !v4 )
      goto LABEL_10;
  }
  else
  {
    v2 = a1 + 8;
  }
  result = 0;
  if ( v2 == *(_QWORD *)(a1 + 24) )
    return result;
  v7 = sub_220EF80(v2);
  v3 = v7 + 32;
  v2 = v7;
LABEL_10:
  v8 = (unsigned __int8)sub_34A0190(v3, a2) == 0;
  result = v2;
  if ( !v8 )
    return 0;
  return result;
}
