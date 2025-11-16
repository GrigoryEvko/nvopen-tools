// Function: sub_1EE9830
// Address: 0x1ee9830
//
__int64 __fastcall sub_1EE9830(__int64 a1, __int64 a2)
{
  __int64 *v4; // rbx
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 result; // rax
  __int64 v10; // r15
  int v11; // r9d
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // r8
  _DWORD *v15; // rcx
  int v16[13]; // [rsp+Ch] [rbp-34h] BYREF

  v4 = (__int64 *)(a1 + 264);
  v5 = *(_QWORD *)(a1 + 8);
  v16[0] = 0;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 200LL))(v5);
  sub_1D05C60((__int64)v4, v6, v16);
  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(v7 + 104);
  result = *(unsigned int *)(v7 + 112);
  v10 = v8 + 8 * result;
  if ( v10 != v8 )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)v8;
      if ( *(int *)v8 >= 0 )
        goto LABEL_3;
      v12 = *(unsigned int *)(a2 + 208);
      v13 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 248) + (v11 & 0x7FFFFFFF));
      if ( v13 < (unsigned int)v12 )
      {
        v14 = *(_QWORD *)(a2 + 200);
        while ( 1 )
        {
          v15 = (_DWORD *)(v14 + 4LL * v13);
          if ( (v11 & 0x7FFFFFFF) == (*v15 & 0x7FFFFFFF) )
            break;
          v13 += 256;
          if ( (unsigned int)v12 <= v13 )
            goto LABEL_10;
        }
        result = v14 + 4 * v12;
        if ( v15 != (_DWORD *)result )
          goto LABEL_3;
      }
LABEL_10:
      result = *(unsigned int *)(v8 + 4);
      if ( (_DWORD)result )
      {
        v8 += 8;
        result = sub_1EE5C30(v4, *(_QWORD **)(a1 + 24), v11);
        if ( v10 == v8 )
          return result;
      }
      else
      {
LABEL_3:
        v8 += 8;
        if ( v10 == v8 )
          return result;
      }
    }
  }
  return result;
}
