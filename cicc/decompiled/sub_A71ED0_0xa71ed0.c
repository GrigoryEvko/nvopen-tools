// Function: sub_A71ED0
// Address: 0xa71ed0
//
__int64 __fastcall sub_A71ED0(__int64 *a1)
{
  int v1; // eax
  char v2; // dl
  __int64 v4; // [rsp+Ch] [rbp-24h]
  __int64 v5; // [rsp+28h] [rbp-8h]

  v1 = sub_A71B70(*a1);
  v2 = 1;
  if ( !v1 )
  {
    HIBYTE(v4) = 0;
    *(_WORD *)((char *)&v4 + 5) = 0;
    v2 = 0;
  }
  LODWORD(v4) = v1;
  BYTE4(v4) = v2;
  v5 = v4;
  BYTE4(v5) = v2;
  return v5;
}
