// Function: sub_82AFD0
// Address: 0x82afd0
//
__int64 __fastcall sub_82AFD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  __int64 v4; // r13
  char v5; // al
  __int64 v6; // rdi
  int *v7; // rdx
  int v8; // r9d
  int v9; // [rsp+Ch] [rbp-24h] BYREF

  v2 = a2;
  for ( result = *(unsigned __int8 *)(a1 + 140); (_BYTE)result == 12; result = *(unsigned __int8 *)(a1 + 140) )
    a1 = *(_QWORD *)(a1 + 160);
  if ( dword_4F077C4 == 2 )
  {
    result = (unsigned int)(result - 9);
    if ( (unsigned __int8)result <= 2u )
    {
      result = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
      v4 = *(_QWORD *)(result + 24);
      if ( v4 )
      {
        if ( dword_4F077BC )
        {
          v5 = *(_BYTE *)(qword_4D03C50 + 17LL);
          if ( (v5 & 0x40) == 0 )
          {
            v6 = *(_QWORD *)(v4 + 88);
            if ( *(char *)(v6 + 192) < 0 )
              goto LABEL_13;
            if ( (*(_BYTE *)(v6 + 195) & 1) != 0 && (unsigned int)sub_736960(v6) )
            {
              v5 = *(_BYTE *)(qword_4D03C50 + 17LL);
LABEL_13:
              if ( (v5 & 2) != 0 )
                sub_8AD0D0(v4, 1, 0);
            }
          }
        }
        if ( (unsigned int)sub_6E6010() && !(unsigned int)sub_884000(v4, 1) )
        {
          v7 = 0;
          v9 = 0;
          if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
            v7 = &v9;
          v8 = 5;
          if ( dword_4D04964 )
            v8 = unk_4F07471;
          sub_87D9B0(v4, 0, 0, a2, 0, v8, unk_4D04460 == 0 ? 2289 : 2867, (__int64)v7);
          if ( v9 )
            sub_6E50A0();
        }
        if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
          v2 = 0;
        result = sub_875C60(v4, 1, v2);
        if ( !(_DWORD)result )
        {
          result = qword_4D03C50;
          if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
            return sub_6E50A0();
        }
      }
    }
  }
  return result;
}
