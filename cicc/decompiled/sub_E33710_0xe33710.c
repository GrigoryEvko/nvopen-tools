// Function: sub_E33710
// Address: 0xe33710
//
__int64 __fastcall sub_E33710(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r9d
  bool v3; // zf
  char v4; // r12
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 result; // rax

  v2 = a2;
  v3 = *(_BYTE *)(a1 + 49) == 0;
  v4 = *(_BYTE *)(a1 + 48);
  *(_BYTE *)(a1 + 48) = 0;
  if ( v3 )
  {
    v5 = *(_QWORD *)(a1 + 40);
    v6 = *(_QWORD *)(a1 + 24);
    if ( v5 < v6 )
    {
      v7 = *(_QWORD *)(a1 + 32);
      if ( *(_BYTE *)(v7 + v5) == 115 )
      {
        *(_QWORD *)(a1 + 40) = v5 + 1;
        if ( v6 > v5 + 1 && *(_BYTE *)(v7 + v5 + 1) == 95 )
        {
          *(_QWORD *)(a1 + 40) = v5 + 2;
        }
        else
        {
          v8 = sub_E31BC0(a1);
          if ( !*(_BYTE *)(a1 + 49) && v8 == -1 )
            *(_BYTE *)(a1 + 49) = 1;
        }
      }
    }
  }
  result = sub_E331B0(a1, v2, 0);
  *(_BYTE *)(a1 + 48) = v4;
  return result;
}
