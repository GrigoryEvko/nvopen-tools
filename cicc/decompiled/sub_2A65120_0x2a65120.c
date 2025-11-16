// Function: sub_2A65120
// Address: 0x2a65120
//
__int64 *__fastcall sub_2A65120(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 *result; // rax
  __int64 v3; // rdx
  __int64 *v4; // r14
  __int64 v5; // rbx
  __int64 *v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // [rsp+8h] [rbp-38h]

  v1 = sub_2A64210(a1);
  result = *(__int64 **)(v1 + 8);
  if ( *(_BYTE *)(v1 + 28) )
    v3 = *(unsigned int *)(v1 + 20);
  else
    v3 = *(unsigned int *)(v1 + 16);
  v4 = &result[v3];
  if ( result != v4 )
  {
    while ( 1 )
    {
      v5 = *result;
      v6 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v4 == ++result )
        return result;
    }
    while ( v4 != v6 )
    {
      v7 = *(_QWORD *)(v5 + 80);
      if ( v7 )
        v7 -= 24;
      if ( (unsigned __int8)sub_2A64220(a1, v7) )
      {
        if ( (*(_BYTE *)(v5 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v5, v7, v8, v9);
          v10 = *(_QWORD *)(v5 + 96);
          v14 = v10 + 40LL * *(_QWORD *)(v5 + 104);
          if ( (*(_BYTE *)(v5 + 2) & 1) != 0 )
          {
            sub_B2C6D0(v5, v7, v12, v13);
            v10 = *(_QWORD *)(v5 + 96);
          }
        }
        else
        {
          v10 = *(_QWORD *)(v5 + 96);
          v14 = v10 + 40LL * *(_QWORD *)(v5 + 104);
        }
        for ( ; v10 != v14; v10 += 40 )
        {
          if ( *(_BYTE *)(*(_QWORD *)(v10 + 8) + 8LL) != 15 )
          {
            v11 = sub_2A64F10((__int64)a1, v10);
            sub_2A61DD0(v5, *(_DWORD *)(v10 + 32) + 1, v11);
          }
        }
      }
      result = v6 + 1;
      if ( v6 + 1 == v4 )
        break;
      while ( 1 )
      {
        v5 = *result;
        v6 = result;
        if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v4 == ++result )
          return result;
      }
    }
  }
  return result;
}
