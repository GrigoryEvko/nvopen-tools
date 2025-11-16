// Function: sub_2044050
// Address: 0x2044050
//
__int64 __fastcall sub_2044050(unsigned int a1, int a2)
{
  __int64 v2; // rbp
  unsigned int v4; // edx
  unsigned int v5; // [rsp-20h] [rbp-20h] BYREF
  unsigned int v6; // [rsp-1Ch] [rbp-1Ch] BYREF
  __int64 v7; // [rsp-8h] [rbp-8h]

  if ( a2 == 0x80000000 )
    return 0;
  v7 = v2;
  v5 = 0x80000000 - a2;
  v4 = sub_16AF780(&v5, 0x80000000);
  if ( a1 > v4 )
    v4 = a1;
  sub_16AF710(&v6, a1, v4);
  return v6;
}
