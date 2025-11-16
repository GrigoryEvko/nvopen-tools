// Function: sub_6E24C0
// Address: 0x6e24c0
//
__int64 sub_6E24C0()
{
  _QWORD *v0; // rdx
  __int64 result; // rax
  _QWORD *v2; // rcx
  __int64 **v3; // rcx
  __int64 *v4; // rdi
  _QWORD *v5; // r8
  __int64 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx

  v0 = (_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  result = qword_4D03C50;
  v2 = *(_QWORD **)(qword_4D03C50 + 72LL);
  v0[39] = v2;
  if ( v2 )
  {
    *v2 = 0;
    result = qword_4D03C50;
  }
  else
  {
    v0[38] = 0;
  }
  v3 = *(__int64 ***)(result + 80);
  if ( v3 )
  {
    v4 = *v3;
    if ( *v3 )
    {
      v5 = (_QWORD *)v0[42];
      v6 = *v3;
      v0[42] = v3;
      *v3 = 0;
      do
      {
        while ( 1 )
        {
          if ( *((_BYTE *)v6 + 16) == 53 )
          {
            v7 = v6[3];
            if ( *(_BYTE *)(v7 + 16) == 6 )
            {
              v8 = *(_QWORD *)(v7 + 24);
              if ( *(__int64 **)(v8 + 96) == v6 )
                break;
            }
          }
          v6 = (__int64 *)*v6;
          if ( !v6 )
            goto LABEL_11;
        }
        *(_QWORD *)(v8 + 96) = 0;
        v6 = (__int64 *)*v6;
      }
      while ( v6 );
LABEL_11:
      *v5 = v0[40];
      result = qword_4D03C50;
      v0[40] = v4;
    }
  }
  *(_BYTE *)(result + 18) &= ~0x20u;
  return result;
}
