// Function: sub_1D274F0
// Address: 0x1d274f0
//
__int64 __fastcall sub_1D274F0(__int128 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __m128i *v9; // r12
  __int128 v11; // [rsp+0h] [rbp-20h] BYREF

  v11 = a1;
  if ( !(_BYTE)a1 )
  {
    if ( qword_4FC15F0 )
    {
      v5 = qword_4FC15F0;
      if ( !(unsigned __int8)sub_16D5D40() )
        goto LABEL_4;
    }
    else
    {
      sub_16C1EA0((__int64)&qword_4FC15F0, sub_160CFB0, (__int64)sub_160D0B0, a3, a4, a5);
      v5 = qword_4FC15F0;
      if ( !(unsigned __int8)sub_16D5D40() )
      {
LABEL_4:
        ++*(_DWORD *)(v5 + 8);
        if ( qword_4FC1630 )
        {
LABEL_5:
          v9 = sub_1D27350((_QWORD *)qword_4FC1630, (const __m128i *)&v11) + 2;
          if ( (unsigned __int8)sub_16D5D40() )
            sub_16C30E0((pthread_mutex_t **)v5);
          else
            --*(_DWORD *)(v5 + 8);
          return (__int64)v9;
        }
LABEL_12:
        sub_16C1EA0((__int64)&qword_4FC1630, sub_1D12F80, (__int64)sub_1D13C60, v6, v7, v8);
        goto LABEL_5;
      }
    }
    sub_16C30C0((pthread_mutex_t **)v5);
    if ( qword_4FC1630 )
      goto LABEL_5;
    goto LABEL_12;
  }
  if ( !qword_4FC1610 )
    sub_16C1EA0((__int64)&qword_4FC1610, sub_1D2EBE0, (__int64)sub_1D13970, a3, a4, a5);
  return *(_QWORD *)qword_4FC1610 + 16LL * (unsigned __int8)a1;
}
