// Function: sub_D472F0
// Address: 0xd472f0
//
__int64 __fastcall sub_D472F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  int v5; // r13d
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // r8
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+20h] [rbp-40h]
  __int64 v16; // [rsp+28h] [rbp-38h]
  __int64 v17; // [rsp+28h] [rbp-38h]

  result = *(_QWORD *)(a1 + 40);
  v14 = result;
  v15 = *(_QWORD *)(a1 + 32);
  if ( result != v15 )
  {
    while ( 1 )
    {
      v3 = *(_QWORD *)(*(_QWORD *)v15 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v3 != *(_QWORD *)v15 + 48LL )
      {
        if ( !v3 )
          BUG();
        v4 = v3 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 <= 0xA )
        {
          v5 = sub_B46E30(v4);
          if ( v5 )
            break;
        }
      }
LABEL_13:
      v15 += 8;
      result = v15;
      if ( v14 == v15 )
        return result;
    }
    v6 = 0;
    while ( 1 )
    {
      v7 = sub_B46EC0(v4, v6);
      v9 = v7;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v10 = *(_QWORD **)(a1 + 64);
        v11 = &v10[*(unsigned int *)(a1 + 76)];
        if ( v10 == v11 )
          goto LABEL_16;
        while ( v9 != *v10 )
        {
          if ( v11 == ++v10 )
            goto LABEL_16;
        }
LABEL_12:
        if ( v5 == ++v6 )
          goto LABEL_13;
      }
      else
      {
        v16 = v7;
        v12 = sub_C8CA60(a1 + 56, v7);
        v9 = v16;
        if ( v12 )
          goto LABEL_12;
LABEL_16:
        v13 = *(unsigned int *)(a2 + 8);
        if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v17 = v9;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 8u, v9, v8);
          v13 = *(unsigned int *)(a2 + 8);
          v9 = v17;
        }
        ++v6;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v13) = v9;
        ++*(_DWORD *)(a2 + 8);
        if ( v5 == v6 )
          goto LABEL_13;
      }
    }
  }
  return result;
}
