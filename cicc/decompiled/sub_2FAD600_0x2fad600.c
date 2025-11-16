// Function: sub_2FAD600
// Address: 0x2fad600
//
unsigned __int64 __fastcall sub_2FAD600(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  char v4; // si
  unsigned __int64 result; // rax
  __int64 v6; // rdx

  if ( (*a1 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v3 = sub_CB59D0(a2, *(unsigned int *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24));
    v4 = aBerd[(*a1 >> 1) & 3];
    result = *(_QWORD *)(v3 + 32);
    if ( result >= *(_QWORD *)(v3 + 24) )
    {
      return sub_CB5D20(v3, v4);
    }
    else
    {
      *(_QWORD *)(v3 + 32) = result + 1;
      *(_BYTE *)result = v4;
    }
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v6) <= 6 )
    {
      return sub_CB6200(a2, (unsigned __int8 *)"invalid", 7u);
    }
    else
    {
      *(_DWORD *)v6 = 1635151465;
      *(_WORD *)(v6 + 4) = 26988;
      *(_BYTE *)(v6 + 6) = 100;
      *(_QWORD *)(a2 + 32) += 7LL;
      return 26988;
    }
  }
  return result;
}
