// Function: sub_1ACD650
// Address: 0x1acd650
//
__int64 __fastcall sub_1ACD650(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-68h]
  __int64 v11; // [rsp+10h] [rbp-60h]
  __int64 v12; // [rsp+20h] [rbp-50h]
  __int64 v13; // [rsp+28h] [rbp-48h]
  char v14[49]; // [rsp+3Fh] [rbp-31h] BYREF

  v11 = a2 + 40;
  v3 = *(_QWORD *)(a2 + 48);
  v13 = *(_QWORD *)(a3 + 48);
  v10 = a3 + 40;
  do
  {
    v4 = 0;
    v5 = 0;
    v14[0] = 1;
    if ( v13 )
      v4 = v13 - 24;
    if ( v3 )
      v5 = v3 - 24;
    result = sub_1ACCFB0(a1, v5, v4, v14);
    if ( (_DWORD)result )
      return result;
    if ( v14[0] && (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) != 0 )
    {
      v7 = 0;
      v12 = 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
        {
          v8 = *(_QWORD *)(*(_QWORD *)(v5 - 8) + v7);
          if ( (*(_BYTE *)(v4 + 23) & 0x40) == 0 )
            goto LABEL_20;
        }
        else
        {
          v8 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) + v7);
          if ( (*(_BYTE *)(v4 + 23) & 0x40) == 0 )
          {
LABEL_20:
            v9 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
            goto LABEL_16;
          }
        }
        v9 = *(_QWORD *)(v4 - 8);
LABEL_16:
        result = sub_1ACCBA0((__int64)a1, v8, *(_QWORD *)(v9 + v7));
        if ( (_DWORD)result )
          return result;
        v7 += 24;
      }
      while ( v12 != v7 );
    }
    v3 = *(_QWORD *)(v3 + 8);
    v13 = *(_QWORD *)(v13 + 8);
    if ( v11 == v3 )
      return (unsigned int)-(v10 != v13);
  }
  while ( v10 != v13 );
  return 1;
}
