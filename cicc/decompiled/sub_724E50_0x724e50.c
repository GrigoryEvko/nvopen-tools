// Function: sub_724E50
// Address: 0x724e50
//
__int64 __fastcall sub_724E50(__int64 *a1, _BYTE *a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9

  if ( dword_4F07270[0] == unk_4F073B8 )
  {
    v2 = *a1;
    if ( (unsigned int)sub_72AA80(*a1) )
    {
      sub_724E30((__int64)a1);
      a2 = 0;
      v2 = sub_740190(v2, 0, 0);
    }
    else
    {
      *a1 = 0;
    }
  }
  else
  {
    a2 = sub_7246D0(208);
    v2 = (__int64)a2;
    sub_72A510(*a1, a2);
    sub_724E30((__int64)a1);
  }
  sub_72A1A0(v2, a2, v3, v4, v5, v6);
  sub_73B910(v2);
  return v2;
}
