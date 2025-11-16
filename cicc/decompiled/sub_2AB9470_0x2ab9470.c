// Function: sub_2AB9470
// Address: 0x2ab9470
//
__int64 __fastcall sub_2AB9470(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 *v6; // r15
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdx
  _BYTE v18[96]; // [rsp+10h] [rbp-60h] BYREF

  v4 = sub_D48970(a1);
  v5 = v4;
  if ( v4 )
    sub_BED950((__int64)v18, a3, v4);
  v6 = *(__int64 **)(a2 + 32);
  result = 11LL * *(unsigned int *)(a2 + 40);
  for ( i = &v6[11 * *(unsigned int *)(a2 + 40)]; i != v6; v6 += 11 )
  {
    v9 = *v6;
    v10 = sub_D47930(a1);
    v11 = *(_QWORD *)(v9 - 8);
    v12 = v10;
    if ( (*(_DWORD *)(v9 + 4) & 0x7FFFFFF) != 0 )
    {
      v13 = 0;
      while ( v12 != *(_QWORD *)(v11 + 32LL * *(unsigned int *)(v9 + 72) + 8 * v13) )
      {
        if ( (*(_DWORD *)(v9 + 4) & 0x7FFFFFF) == (_DWORD)++v13 )
          goto LABEL_16;
      }
      v14 = 32 * v13;
    }
    else
    {
LABEL_16:
      v14 = 0x1FFFFFFFE0LL;
    }
    v15 = *(_QWORD *)(v11 + v14);
    result = *(_QWORD *)(v15 + 16);
    if ( result )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(result + 24);
        if ( v9 != v16 && v5 != v16 )
          break;
        result = *(_QWORD *)(result + 8);
        if ( !result )
          goto LABEL_15;
      }
    }
    else
    {
LABEL_15:
      result = (__int64)sub_AE6EC0(a3, v15);
    }
  }
  return result;
}
