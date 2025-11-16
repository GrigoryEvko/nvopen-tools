// Function: sub_39B1910
// Address: 0x39b1910
//
__int64 __fastcall sub_39B1910(__int64 a1, _QWORD *a2, unsigned int a3, unsigned int a4, unsigned int a5, __int64 a6)
{
  unsigned int v6; // r10d
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 (*v13)(); // rax

  if ( a3 == 32 )
  {
    LOBYTE(v10) = 5;
    goto LABEL_5;
  }
  if ( a3 <= 0x20 )
  {
    if ( a3 == 8 )
    {
      LOBYTE(v10) = 3;
    }
    else
    {
      LOBYTE(v10) = 4;
      if ( a3 != 16 )
      {
        LOBYTE(v10) = 2;
        if ( a3 != 1 )
          goto LABEL_9;
      }
    }
LABEL_5:
    v11 = 0;
    goto LABEL_6;
  }
  if ( a3 == 64 )
  {
    LOBYTE(v10) = 6;
    goto LABEL_5;
  }
  if ( a3 == 128 )
  {
    LOBYTE(v10) = 7;
    goto LABEL_5;
  }
LABEL_9:
  v10 = sub_1F58CC0(a2, a3);
  v6 = v10;
LABEL_6:
  v12 = *(_QWORD *)(a1 + 24);
  LOBYTE(v6) = v10;
  v13 = *(__int64 (**)())(*(_QWORD *)v12 + 448LL);
  if ( v13 == sub_1D12D60 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD, __int64))v13)(v12, v6, v11, a4, a5, a6);
}
