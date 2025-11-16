// Function: sub_B44AB0
// Address: 0xb44ab0
//
__int64 __fastcall sub_B44AB0(unsigned __int8 *a1)
{
  __int64 v1; // rbp
  __int64 v3; // rdx
  __int64 v5; // [rsp-28h] [rbp-28h] BYREF
  _QWORD v6[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( (unsigned __int8)(*a1 - 34) > 0x33u )
    return 0;
  v3 = 0x8000000000041LL;
  if ( !_bittest64(&v3, (unsigned int)*a1 - 34) )
    return 0;
  v6[3] = v1;
  v6[0] = *((_QWORD *)a1 + 9);
  v5 = sub_A74610(v6);
  if ( (unsigned __int8)sub_A73170(&v5, 97) || (unsigned __int8)sub_A73170(&v5, 86) )
    return 1;
  else
    return (unsigned int)sub_A73170(&v5, 43);
}
