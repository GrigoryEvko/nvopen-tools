// Function: sub_30D4E50
// Address: 0x30d4e50
//
__int64 __fastcall sub_30D4E50(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v4; // [rsp+8h] [rbp-8h] BYREF

  if ( *a1 && (v1 = sub_A72240(a1), !sub_C93CC0(v1, v2, 0xAu, &v4)) && v4 == (int)v4 )
  {
    BYTE4(v4) = 1;
    return v4;
  }
  else
  {
    BYTE4(v4) = 0;
    return v4;
  }
}
