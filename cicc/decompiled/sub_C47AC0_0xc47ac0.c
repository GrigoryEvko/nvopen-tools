// Function: sub_C47AC0
// Address: 0xc47ac0
//
__int64 __fastcall sub_C47AC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  unsigned int v3; // r14d
  unsigned int v4; // r13d
  _QWORD *v5; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // rax

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
        sub_C47690((__int64 *)a1, v4);
        return a1;
      }
      goto LABEL_6;
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
    v7 = *(_QWORD *)a1 << (char)v5;
    goto LABEL_7;
  }
LABEL_6:
  v7 = 0;
LABEL_7:
  v8 = v7 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v4);
  if ( !v4 )
    v8 = 0;
  *(_QWORD *)a1 = v8;
  return a1;
}
