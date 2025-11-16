// Function: sub_7E1F40
// Address: 0x7e1f40
//
__int64 __fastcall sub_7E1F40(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3; // rax

  v1 = sub_7E1E20(a1);
  result = sub_8D3D10(v1);
  if ( (_DWORD)result )
  {
    v3 = sub_8D4870(v1);
    return (unsigned int)sub_8D2310(v3) == 0;
  }
  return result;
}
