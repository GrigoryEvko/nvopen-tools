// Function: sub_8CA420
// Address: 0x8ca420
//
__int64 __fastcall sub_8CA420(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  __int64 v3; // r12
  __int64 v4; // rbx

  result = sub_8CA330(a1);
  if ( result && a1 != result )
  {
    v2 = **(_BYTE **)(result + 176) & 1;
    if ( (**(_BYTE **)(a1 + 176) & 1) != 0 )
    {
      v4 = *(_QWORD *)(a1 + 168);
      if ( (*(_BYTE *)(a1 + 161) & 0x10) != 0 )
        v4 = *(_QWORD *)(v4 + 96);
      if ( !v2 )
        goto LABEL_16;
      v3 = *(_QWORD *)(result + 168);
      if ( (*(_BYTE *)(result + 161) & 0x10) != 0 )
        goto LABEL_6;
LABEL_9:
      while ( v4 )
      {
        if ( !v3 )
        {
          do
          {
            result = (__int64)sub_8C7090(2, v4);
            v4 = *(_QWORD *)(v4 + 120);
LABEL_16:
            ;
          }
          while ( v4 );
          return result;
        }
        result = sub_8CBB20(2, v4, v3);
        v3 = *(_QWORD *)(v3 + 120);
        v4 = *(_QWORD *)(v4 + 120);
      }
    }
    else if ( v2 )
    {
      v3 = *(_QWORD *)(result + 168);
      v4 = 0;
      if ( (*(_BYTE *)(result + 161) & 0x10) != 0 )
      {
LABEL_6:
        v3 = *(_QWORD *)(v3 + 96);
        goto LABEL_9;
      }
    }
  }
  return result;
}
