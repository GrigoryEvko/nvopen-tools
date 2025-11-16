// Function: sub_892880
// Address: 0x892880
//
__int64 __fastcall sub_892880(__int64 a1)
{
  char v1; // dl
  __int64 v2; // r8
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  v1 = *(_BYTE *)(a1 + 80);
  v2 = *(_QWORD *)(a1 + 88);
  if ( v1 == 20 )
  {
    v5 = *(_QWORD *)(v2 + 88);
    if ( !v5 || (*(_BYTE *)(v2 + 160) & 1) != 0 )
      result = v2 + 296;
    else
      result = *(_QWORD *)(v5 + 88) + 296LL;
    if ( !*(_QWORD *)(result + 32) )
      return sub_892400(*(_QWORD *)(a1 + 88));
  }
  else
  {
    result = *(_QWORD *)(a1 + 88);
    if ( v1 == 21 )
    {
      v4 = *(_QWORD *)(v2 + 88);
      if ( !v4 || (*(_BYTE *)(v2 + 160) & 1) != 0 )
        result = v2 + 200;
      else
        result = *(_QWORD *)(v4 + 88) + 200LL;
      if ( !*(_QWORD *)(result + 32) )
      {
        if ( *(_QWORD *)(v2 + 32) )
          return *(_QWORD *)(a1 + 88);
      }
    }
  }
  return result;
}
