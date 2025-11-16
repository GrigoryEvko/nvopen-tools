// Function: sub_2AA7EC0
// Address: 0x2aa7ec0
//
__int64 __fastcall sub_2AA7EC0(__int64 a1, char *a2, char a3)
{
  int v3; // ebx
  int v5; // eax
  bool v7; // al
  __int64 v8; // [rsp+8h] [rbp-28h]

  v5 = sub_DCF980(*(__int64 **)(a1 + 112), a2);
  if ( v5 )
  {
    LODWORD(v8) = v5;
    BYTE4(v8) = 1;
  }
  else
  {
    if ( byte_500DBA8 && (v8 = sub_F6EC60((__int64)a2, 0), BYTE4(v8)) )
    {
      v3 = v8;
      v7 = 1;
    }
    else
    {
      v7 = 0;
      if ( a3 )
      {
        v3 = sub_DEF800(a1);
        if ( v3 )
          v7 = 1;
      }
    }
    LODWORD(v8) = v3;
    BYTE4(v8) = v7;
  }
  return v8;
}
