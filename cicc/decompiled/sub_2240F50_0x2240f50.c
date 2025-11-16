// Function: sub_2240F50
// Address: 0x2240f50
//
unsigned __int64 __fastcall sub_2240F50(unsigned __int64 *a1, char a2)
{
  unsigned __int64 v2; // rbp
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdx
  unsigned __int64 result; // rax

  v2 = a1[1];
  v3 = *a1;
  v4 = v2 + 1;
  if ( (unsigned __int64 *)*a1 == a1 + 2 )
    v5 = 15;
  else
    v5 = a1[2];
  if ( v4 > v5 )
  {
    sub_2240BB0(a1, a1[1], 0, 0, 1u);
    v3 = *a1;
  }
  *(_BYTE *)(v3 + v2) = a2;
  result = *a1;
  a1[1] = v4;
  *(_BYTE *)(result + v2 + 1) = 0;
  return result;
}
