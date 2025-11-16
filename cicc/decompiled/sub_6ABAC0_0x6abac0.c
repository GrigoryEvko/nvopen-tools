// Function: sub_6ABAC0
// Address: 0x6abac0
//
__int64 __fastcall sub_6ABAC0(__int64 *a1, unsigned int *a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 i; // r12
  unsigned int v6; // r13d
  __int64 v7; // rdi
  char v8; // al
  char v9; // dl

  result = sub_6AB980();
  v4 = *(_BYTE *)(result + 140);
  for ( i = result; v4 == 12; v4 = *(_BYTE *)(result + 140) )
    result = *(_QWORD *)(result + 160);
  if ( v4 )
  {
    v6 = *a2;
    if ( !*a2 )
    {
      v7 = *a1;
      if ( i != *a1 )
      {
        if ( !v7 || !dword_4F07588 || (result = *(_QWORD *)(i + 32), *(_QWORD *)(v7 + 32) != result) || !result )
        {
          v8 = *(_BYTE *)(v7 + 160);
          if ( v8 == 6 || (v9 = *(_BYTE *)(i + 160), v9 == 6) )
          {
            v6 = 6;
          }
          else if ( v9 == 4 || v8 == 4 )
          {
            v6 = 4;
          }
          else if ( v8 == 2 || v9 == 2 )
          {
            v6 = 2;
          }
          else if ( v9 && v8 )
          {
            if ( v9 != 1 && v8 != 1 )
              sub_721090(v7);
            v6 = 1;
          }
          if ( (unsigned int)sub_8D2B50(v7) || (unsigned int)sub_8D2B50(i) )
          {
            result = sub_72C6F0(v6);
            *a1 = result;
          }
          else
          {
            result = sub_72C610(v6);
            *a1 = result;
          }
        }
      }
    }
  }
  else
  {
    *a2 = 1;
  }
  return result;
}
