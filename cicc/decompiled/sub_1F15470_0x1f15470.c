// Function: sub_1F15470
// Address: 0x1f15470
//
__int64 __fastcall sub_1F15470(_QWORD *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 result; // rax
  int v11; // esi
  unsigned int v12; // r13d
  unsigned __int16 v13; // dx
  __int64 j; // rbx
  __int64 i; // rbx
  __int64 *v16; // rdx

  v5 = *(_QWORD *)(a2 + 104);
  if ( !v5 )
    return sub_1DB7A00((__int64 *)a2, a3);
  v6 = *(_QWORD *)(a3 + 8);
  if ( a4 )
  {
    do
    {
      for ( i = *(_QWORD *)(*(_QWORD *)(a1[9] + 8LL) + 104LL);
            *(_DWORD *)(i + 112) != *(_DWORD *)(v5 + 112);
            i = *(_QWORD *)(i + 104) )
      {
        ;
      }
      v16 = (__int64 *)sub_1DB3C70((__int64 *)i, v6);
      result = *(_QWORD *)i + 24LL * *(unsigned int *)(i + 8);
      if ( v16 != (__int64 *)result )
      {
        result = *(_DWORD *)((*v16 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v16 >> 1) & 3;
        if ( (unsigned int)result <= (*(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6 >> 1) & 3) )
        {
          result = v16[2];
          if ( result )
          {
            if ( *(_QWORD *)(result + 8) == v6 )
              result = sub_1DB79D0((__int64 *)v5, v6, (__int64 *)(a1[2] + 296LL));
          }
        }
      }
      v5 = *(_QWORD *)(v5 + 104);
    }
    while ( v5 );
    return result;
  }
  if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    BUG();
  v7 = *(_QWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  v8 = *(_QWORD *)(v7 + 32);
  v9 = v8 + 40LL * (unsigned int)sub_1E163A0(v7);
  result = *(_QWORD *)(v7 + 32);
  if ( v9 == result )
  {
    v12 = 0;
    goto LABEL_10;
  }
  v11 = *(_DWORD *)(a2 + 112);
  v12 = 0;
  while ( v11 != *(_DWORD *)(result + 8) )
  {
LABEL_9:
    result += 40;
    if ( result == v9 )
      goto LABEL_10;
  }
  v13 = (*(_DWORD *)result >> 8) & 0xFFF;
  if ( v13 )
  {
    v12 |= *(_DWORD *)(*(_QWORD *)(a1[7] + 248LL) + 4LL * v13);
    goto LABEL_9;
  }
  result = sub_1E69F40(a1[4], v11);
  v12 = result;
LABEL_10:
  for ( j = *(_QWORD *)(a2 + 104); j; j = *(_QWORD *)(j + 104) )
  {
    while ( 1 )
    {
      result = *(_DWORD *)(j + 112) & v12;
      if ( (*(_DWORD *)(j + 112) & v12) != 0 )
        break;
      j = *(_QWORD *)(j + 104);
      if ( !j )
        return result;
    }
    result = sub_1DB79D0((__int64 *)j, v6, (__int64 *)(a1[2] + 296LL));
  }
  return result;
}
