// Function: sub_33EC210
// Address: 0x33ec210
//
__int64 *__fastcall sub_33EC210(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *result; // rax
  unsigned int v11; // edx
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // [rsp+8h] [rbp-28h] BYREF

  v7 = a3;
  v8 = a3 + 16 * a4;
  v9 = a2[5];
  if ( v8 == a3 )
    return a2;
  while ( *(_QWORD *)a3 == *(_QWORD *)v9 && *(_DWORD *)(a3 + 8) == *(_DWORD *)(v9 + 8) )
  {
    a3 += 16;
    v9 += 40;
    if ( v8 == a3 )
      return a2;
  }
  v19 = 0;
  result = sub_33E34E0((__int64)a1, (__int64)a2, v7, a4, (__int64 *)&v19);
  if ( !result )
  {
    if ( v19 && !(unsigned __int8)sub_33EB970((__int64)a1, (__int64)a2, v11) )
      v19 = 0;
    if ( (_DWORD)a4 )
    {
      v12 = v7;
      v13 = 0;
      v14 = v7 + 16LL * (unsigned int)(a4 - 1) + 16;
      do
      {
        while ( 1 )
        {
          v18 = v13 + a2[5];
          if ( *(_QWORD *)v18 != *(_QWORD *)v12 || *(_DWORD *)(v18 + 8) != *(_DWORD *)(v12 + 8) )
            break;
          v12 += 16;
          v13 += 40;
          if ( v12 == v14 )
            goto LABEL_20;
        }
        if ( *(_QWORD *)v18 )
        {
          v15 = *(_QWORD *)(v18 + 32);
          **(_QWORD **)(v18 + 24) = v15;
          if ( v15 )
            *(_QWORD *)(v15 + 24) = *(_QWORD *)(v18 + 24);
        }
        *(_QWORD *)v18 = *(_QWORD *)v12;
        *(_DWORD *)(v18 + 8) = *(_DWORD *)(v12 + 8);
        v16 = *(_QWORD *)v12;
        if ( *(_QWORD *)v12 )
        {
          v17 = *(_QWORD *)(v16 + 56);
          *(_QWORD *)(v18 + 32) = v17;
          if ( v17 )
            *(_QWORD *)(v17 + 24) = v18 + 32;
          *(_QWORD *)(v18 + 24) = v16 + 56;
          *(_QWORD *)(v16 + 56) = v18;
        }
        v12 += 16;
        v13 += 40;
      }
      while ( v12 != v14 );
    }
LABEL_20:
    sub_33CEF80(a1, (__int64)a2);
    if ( v19 )
      sub_C657C0(a1 + 65, a2, v19, (__int64)off_4A367D0);
    return a2;
  }
  return result;
}
