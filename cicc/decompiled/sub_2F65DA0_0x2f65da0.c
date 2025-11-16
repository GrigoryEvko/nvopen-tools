// Function: sub_2F65DA0
// Address: 0x2f65da0
//
__int64 __fastcall sub_2F65DA0(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r12
  char v4; // r15
  _QWORD *v5; // rax
  __int64 v6; // rax

  v2 = *(_QWORD *)(a1[53] + 8LL * a2);
  if ( !v2 )
  {
    v4 = qword_501EA48[8];
    v5 = (_QWORD *)sub_22077B0(0x68u);
    v2 = (__int64)v5;
    if ( v5 )
    {
      *v5 = v5 + 2;
      v5[1] = 0x200000000LL;
      v5[8] = v5 + 10;
      v5[9] = 0x200000000LL;
      if ( v4 )
      {
        v6 = sub_22077B0(0x30u);
        if ( v6 )
        {
          *(_DWORD *)(v6 + 8) = 0;
          *(_QWORD *)(v6 + 16) = 0;
          *(_QWORD *)(v6 + 24) = v6 + 8;
          *(_QWORD *)(v6 + 32) = v6 + 8;
          *(_QWORD *)(v6 + 40) = 0;
        }
        *(_QWORD *)(v2 + 96) = v6;
      }
      else
      {
        v5[12] = 0;
      }
    }
    *(_QWORD *)(a1[53] + 8LL * a2) = v2;
    sub_2E11710(a1, v2, a2);
  }
  return v2;
}
