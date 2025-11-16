// Function: sub_12BAB40
// Address: 0x12bab40
//
__int64 __fastcall sub_12BAB40(__int64 a1, int a2, __int64 a3)
{
  char v4; // r13
  __int64 v5; // r15
  __int64 *v6; // rcx
  __int64 *v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // rdx

  v4 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v5 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( a1 )
    {
      v6 = *(__int64 **)(a1 + 192);
      v7 = *(__int64 **)(a1 + 184);
      if ( a2 != v6 - v7 )
      {
        v8 = 4;
        sub_16C30E0(v5);
        return v8;
      }
      v4 = 1;
      v8 = 0;
      if ( v7 != v6 )
        goto LABEL_8;
    }
    else
    {
      v8 = 5;
    }
    goto LABEL_18;
  }
  if ( !qword_4F92D80 )
    sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
  v5 = qword_4F92D80;
  if ( !a1 )
    return 5;
  v6 = *(__int64 **)(a1 + 192);
  v7 = *(__int64 **)(a1 + 184);
  v8 = 4;
  if ( v6 - v7 == a2 )
  {
    v8 = 0;
    if ( v6 != v7 )
    {
LABEL_8:
      while ( a3 )
      {
        v9 = *v7++;
        a3 += 8;
        *(_QWORD *)(a3 - 8) = v9;
        if ( v6 == v7 )
        {
          v8 = 0;
          goto LABEL_11;
        }
      }
      v8 = 4;
LABEL_11:
      if ( !v4 )
        return v8;
LABEL_18:
      sub_16C30E0(v5);
    }
  }
  return v8;
}
