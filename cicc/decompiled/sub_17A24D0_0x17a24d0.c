// Function: sub_17A24D0
// Address: 0x17a24d0
//
void __fastcall sub_17A24D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __int64 v5; // rax

  if ( *(_DWORD *)(a1 + 8) <= 0x40u && *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    v2 = *(_QWORD *)a2;
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v3 = *(unsigned int *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v3;
    v4 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
    if ( (unsigned int)v3 > 0x40 )
    {
      v5 = (unsigned int)((unsigned __int64)(v3 + 63) >> 6) - 1;
      *(_QWORD *)(v2 + 8 * v5) &= v4;
    }
    else
    {
      *(_QWORD *)a1 = v4 & v2;
    }
  }
  else
  {
    sub_16A51C0(a1, a2);
  }
}
