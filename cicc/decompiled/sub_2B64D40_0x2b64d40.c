// Function: sub_2B64D40
// Address: 0x2b64d40
//
__int64 __fastcall sub_2B64D40(
        unsigned __int8 *a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        unsigned __int8 *a4,
        __int64 *a5)
{
  unsigned int v5; // r14d
  unsigned __int8 *v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v15[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_2B0D8B0(a1) && (unsigned __int8)sub_2B0D8B0(a3) )
    return 1;
  if ( (unsigned __int8)sub_2B0D8B0(a2) && (unsigned __int8)sub_2B0D8B0(a4) )
    return 1;
  if ( *v11 <= 0x1Cu && *a3 <= 0x1Cu && *a2 <= 0x1Cu && *a4 <= 0x1Cu )
    return 1;
  LOBYTE(v5) = a2 == a4 || v11 == a3;
  if ( (_BYTE)v5 )
    return 1;
  v14[1] = (__int64)a3;
  v14[0] = (__int64)v11;
  if ( sub_2B5F980(v14, 2u, a5) && v12 )
  {
    return 1;
  }
  else
  {
    v15[0] = (__int64)a2;
    v15[1] = (__int64)a4;
    if ( sub_2B5F980(v15, 2u, a5) )
      LOBYTE(v5) = v13 != 0;
  }
  return v5;
}
