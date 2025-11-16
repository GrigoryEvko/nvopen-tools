// Function: sub_CB6CE0
// Address: 0xcb6ce0
//
__int64 __fastcall sub_CB6CE0(__int64 *a1)
{
  unsigned int v1; // r13d
  unsigned int v3; // eax

  v1 = *((unsigned __int8 *)a1 + 40);
  if ( !(_BYTE)v1 || (unsigned __int8)sub_C86400() && !(*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 48))(a1) )
    return 0;
  v3 = sub_C86400();
  if ( !(_BYTE)v3 )
    return v1;
  v1 = v3;
  if ( a1[4] == a1[2] )
    return v1;
  sub_CB5AE0(a1);
  return v1;
}
