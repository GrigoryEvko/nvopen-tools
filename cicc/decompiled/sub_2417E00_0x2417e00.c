// Function: sub_2417E00
// Address: 0x2417e00
//
_QWORD *__fastcall sub_2417E00(__int64 **a1, __int64 a2)
{
  __int64 *v3; // r14
  unsigned int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 **v7; // rax
  __int64 v8; // r12
  _QWORD *result; // rax
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 v12; // r12
  __int64 *v13; // rdx
  char *v14; // r12
  __int64 v15; // r13
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *a1;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v4 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v13 = *(__int64 **)(a2 - 8);
    else
      v13 = (__int64 *)(a2 - 32LL * v4);
    v14 = (char *)sub_24159D0((__int64)v3, *v13);
    v15 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v22 = a2 + 24;
    if ( (unsigned int)v15 > 1 )
    {
      v21 = 32 * v15;
      v16 = 32;
      do
      {
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v17 = *(_QWORD *)(a2 - 8);
        else
          v17 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v18 = *(_QWORD *)(v17 + v16);
        v16 += 32;
        v19 = sub_24159D0((__int64)v3, v18);
        v14 = (char *)sub_2416F70((__int64)v3, (unsigned __int64)v14, v19, v22, 0);
      }
      while ( v16 != v21 );
    }
    v8 = sub_2414F50((__int64)v3, *(_QWORD *)(a2 + 8), v14, v22, 0);
    v3 = *a1;
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = *v3;
    if ( (unsigned __int8)(*(_BYTE *)(v5 + 8) - 15) > 1u )
    {
      v8 = *(_QWORD *)(v6 + 72);
    }
    else
    {
      v7 = (__int64 **)sub_240F000(v6, v5);
      v8 = sub_AC9350(v7);
      v3 = *a1;
    }
  }
  v23[0] = a2;
  *sub_FAA780((__int64)(v3 + 22), v23) = v8;
  result = (_QWORD *)sub_240D530();
  if ( (_BYTE)result )
  {
    v10 = sub_2416BC0(*a1, a2);
    v11 = *a1;
    v12 = v10;
    result = (_QWORD *)sub_240D530();
    if ( (_BYTE)result )
    {
      v23[0] = a2;
      result = sub_FAA780((__int64)(v11 + 26), v23);
      *result = v12;
    }
  }
  return result;
}
