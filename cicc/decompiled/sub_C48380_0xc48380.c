// Function: sub_C48380
// Address: 0xc48380
//
void __fastcall sub_C48380(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  unsigned int v3; // r14d
  unsigned int v4; // r13d
  _QWORD *v5; // rcx

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(_DWORD *)(a2 + 8);
  v4 = *(_DWORD *)(a1 + 8);
  if ( v3 > 0x40 )
  {
    if ( v3 - (unsigned int)sub_C444A0(a2) > 0x40 || (v5 = **(_QWORD ***)a2, v2 < (unsigned __int64)v5) )
    {
LABEL_3:
      if ( v4 > 0x40 )
      {
LABEL_4:
        sub_C482E0(a1, v4);
        return;
      }
      goto LABEL_10;
    }
  }
  else
  {
    v5 = *(_QWORD **)a2;
    if ( v2 < *(_QWORD *)a2 )
      goto LABEL_3;
  }
  if ( v4 > 0x40 )
  {
    v4 = (unsigned int)v5;
    goto LABEL_4;
  }
  if ( v4 != (_DWORD)v5 )
  {
    *(_QWORD *)a1 >>= (char)v5;
    return;
  }
LABEL_10:
  *(_QWORD *)a1 = 0;
}
