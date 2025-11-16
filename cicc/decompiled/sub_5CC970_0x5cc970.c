// Function: sub_5CC970
// Address: 0x5cc970
//
__int64 __fastcall sub_5CC970(char a1)
{
  __int64 result; // rax
  __int64 *v2; // r12
  __int64 v3; // rax
  __int64 v4; // [rsp+8h] [rbp-28h] BYREF

  result = 0;
  v4 = 0;
  if ( unk_4D043E0 && word_4F06418[0] == 142 )
  {
    v2 = &v4;
    while ( 1 )
    {
      v3 = sub_5CC040(a1);
      *v2 = v3;
      if ( word_4F06418[0] != 142 )
        break;
      if ( v3 )
        v2 = (__int64 *)sub_5CB9F0((_QWORD **)v2);
    }
    return v4;
  }
  return result;
}
