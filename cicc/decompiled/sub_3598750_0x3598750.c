// Function: sub_3598750
// Address: 0x3598750
//
__int64 __fastcall sub_3598750(__int64 a1, _QWORD *a2, __int64 a3, char a4)
{
  __int64 result; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  char v12; // [rsp+17h] [rbp-39h]

  do
  {
    result = sub_2E311E0(a1);
    v6 = *(_QWORD *)(a1 + 56);
    v7 = result;
    if ( v6 == result )
      return result;
    v8 = 0;
    while ( 1 )
    {
      if ( !v6 )
        BUG();
      v9 = v6;
      if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 8) != 0 )
      {
        do
          v9 = *(_QWORD *)(v9 + 8);
        while ( (*(_BYTE *)(v9 + 44) & 8) != 0 );
      }
      v10 = *(_QWORD *)(v9 + 8);
      v11 = *(unsigned int *)(*(_QWORD *)(v6 + 32) + 8LL);
      if ( (int)v11 < 0 )
        result = *(_QWORD *)(a2[7] + 16 * (v11 & 0x7FFFFFFF) + 8);
      else
        result = *(_QWORD *)(a2[38] + 8 * v11);
      if ( !result )
        goto LABEL_16;
      if ( (*(_BYTE *)(result + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          result = *(_QWORD *)(result + 32);
          if ( !result )
            break;
          if ( (*(_BYTE *)(result + 3) & 0x10) == 0 )
            goto LABEL_11;
        }
LABEL_16:
        if ( !a3 )
          goto LABEL_18;
LABEL_17:
        sub_2FAD510(*(_QWORD *)(a3 + 32), v6);
        goto LABEL_18;
      }
LABEL_11:
      if ( !a4 )
      {
        v12 = v8;
        result = sub_2E88F80(v6);
        v8 = v12;
        if ( (_DWORD)result == 3 )
          break;
      }
      if ( v7 == v10 )
        goto LABEL_19;
LABEL_13:
      v6 = v10;
    }
    sub_2EBE590(
      (__int64)a2,
      *(_DWORD *)(*(_QWORD *)(v6 + 32) + 48LL),
      *(_QWORD *)(a2[7] + 16LL * (*(_DWORD *)(*(_QWORD *)(v6 + 32) + 8LL) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
      0);
    sub_2EBECB0(a2, *(_DWORD *)(*(_QWORD *)(v6 + 32) + 8LL), *(_DWORD *)(*(_QWORD *)(v6 + 32) + 48LL));
    if ( a3 )
      goto LABEL_17;
LABEL_18:
    result = sub_2E88E20(v6);
    v8 = 1;
    if ( v7 != v10 )
      goto LABEL_13;
LABEL_19:
    ;
  }
  while ( v8 );
  return result;
}
