// Function: sub_11B1B00
// Address: 0x11b1b00
//
__int64 __fastcall sub_11B1B00(_QWORD **a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // r14d
  _QWORD *v4; // rax
  __int64 v5; // rdx

  v2 = 0;
  if ( *(_BYTE *)a2 == 17 )
  {
    v3 = *(_DWORD *)(a2 + 32);
    if ( v3 <= 0x40 )
    {
      v4 = *a1;
      v5 = *(_QWORD *)(a2 + 24);
      goto LABEL_5;
    }
    if ( v3 - (unsigned int)sub_C444A0(a2 + 24) <= 0x40 )
    {
      v4 = *a1;
      v5 = **(_QWORD **)(a2 + 24);
LABEL_5:
      *v4 = v5;
      return 1;
    }
  }
  return v2;
}
