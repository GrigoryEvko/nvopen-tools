// Function: sub_3249F00
// Address: 0x3249f00
//
__int64 __fastcall sub_3249F00(__int64 *a1, __int64 a2, char a3)
{
  int v3; // edx
  __int64 result; // rax

  v3 = a3 & 3;
  switch ( v3 )
  {
    case 2:
      return sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 50, 65547, 2);
    case 1:
      return sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 50, 65547, 3);
    case 3:
      return sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 50, 65547, 1);
  }
  return result;
}
