// Function: sub_135CE70
// Address: 0x135ce70
//
__int64 __fastcall sub_135CE70(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 result; // rax

  v2 = a2 + 40;
  v3 = *(_QWORD *)(a2 + 48);
  if ( v3 != a2 + 40 )
  {
    do
    {
      v4 = v3 - 24;
      if ( !v3 )
        v4 = 0;
      result = sub_135CDE0(a1, v4);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
  return result;
}
