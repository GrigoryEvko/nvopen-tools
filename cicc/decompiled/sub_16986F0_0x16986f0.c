// Function: sub_16986F0
// Address: 0x16986f0
//
void *__fastcall sub_16986F0(_BYTE *a1, char a2, char a3, __int64 *a4)
{
  __int64 v6; // r12
  unsigned int v7; // eax
  unsigned int v8; // r13d
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 *v12; // rsi
  __int64 v13; // rsi
  unsigned int v14; // eax
  __int64 v15; // rdx
  int v16; // r15d
  __int64 v17; // rsi
  void *result; // rax

  a1[18] = a1[18] & 0xF0 | (8 * a3 + 1) & 0xF;
  v6 = sub_1698470((__int64)a1);
  v7 = sub_1698310((__int64)a1);
  v8 = v7;
  if ( a4 )
  {
    v9 = *((unsigned int *)a4 + 2);
    v10 = (unsigned __int64)(v9 + 63) >> 6;
    v11 = (unsigned int)v10;
    if ( v8 > (unsigned int)v10 )
    {
      sub_16A7020(v6, 0, v8);
      v9 = *((unsigned int *)a4 + 2);
      v11 = (unsigned __int64)(v9 + 63) >> 6;
    }
    v12 = a4;
    if ( v8 <= (unsigned int)v11 )
      v11 = v8;
    if ( (unsigned int)v9 > 0x40 )
      v12 = (__int64 *)*a4;
    sub_16A7050(v6, v12, v11);
    v13 = (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 4LL) - 1) >> 6;
    v14 = v13 + 1;
    *(_QWORD *)(v6 + 8 * v13) &= ~(-1LL << (*(_BYTE *)(*(_QWORD *)a1 + 4LL) - 1));
    if ( v8 != (_DWORD)v13 + 1 )
    {
      do
      {
        v15 = v14++;
        *(_QWORD *)(v6 + 8 * v15) = 0;
      }
      while ( v8 != v14 );
    }
  }
  else
  {
    sub_16A7020(v6, 0, v7);
  }
  v16 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  v17 = (unsigned int)(v16 - 2);
  if ( !a2 )
  {
    sub_16A70D0(v6, v17);
    result = &unk_42AE9B0;
    if ( *(_UNKNOWN **)a1 != &unk_42AE9B0 )
      return result;
    return (void *)sub_16A70D0(v6, (unsigned int)(v16 - 1));
  }
  sub_16A70F0(v6, v17);
  if ( (unsigned __int8)sub_16A7080(v6, v8) )
    sub_16A70D0(v6, (unsigned int)(v16 - 3));
  result = &unk_42AE9B0;
  if ( *(_UNKNOWN **)a1 == &unk_42AE9B0 )
    return (void *)sub_16A70D0(v6, (unsigned int)(v16 - 1));
  return result;
}
