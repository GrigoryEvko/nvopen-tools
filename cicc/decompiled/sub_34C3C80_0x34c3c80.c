// Function: sub_34C3C80
// Address: 0x34c3c80
//
unsigned __int64 __fastcall sub_34C3C80(unsigned __int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned __int64 result; // rax
  unsigned __int8 *v5; // r8

  v3 = a1[1];
  if ( v3 == a1[2] )
    return sub_34C3A40(a1, v3, a2);
  if ( v3 )
  {
    *(_DWORD *)v3 = *(_DWORD *)a2;
    result = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(v3 + 8) = result;
    v5 = *(unsigned __int8 **)(a2 + 16);
    *(_QWORD *)(v3 + 16) = v5;
    if ( v5 )
    {
      result = sub_B976B0(a2 + 16, v5, v3 + 16);
      *(_QWORD *)(a2 + 16) = 0;
    }
    v3 = a1[1];
  }
  a1[1] = v3 + 24;
  return result;
}
