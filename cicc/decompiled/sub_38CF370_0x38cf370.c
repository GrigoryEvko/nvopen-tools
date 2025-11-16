// Function: sub_38CF370
// Address: 0x38cf370
//
__int64 __fastcall sub_38CF370(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r15
  __int64 v5; // r14
  int v6; // r8d
  int v7; // r9d
  _QWORD *v8; // r12
  _QWORD *i; // r13
  __int64 v10; // r15
  int v11; // r8d
  int v12; // r9d
  const void *v13; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 + 8) = a1 + 24;
  v13 = (const void *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0x1000000000LL;
  result = a1 + 8;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 176) = 0;
  v3 = *(_QWORD **)(a2 + 40);
  v4 = *(_QWORD **)(a2 + 32);
  if ( v4 != v3 )
  {
    do
    {
      while ( 1 )
      {
        v5 = *v4;
        result = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 16LL))(*v4);
        if ( !(_BYTE)result )
          break;
        if ( v3 == ++v4 )
          goto LABEL_8;
      }
      result = *(unsigned int *)(a1 + 16);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 20) )
      {
        sub_16CD150(a1 + 8, v13, 0, 8, v6, v7);
        result = *(unsigned int *)(a1 + 16);
      }
      ++v4;
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * result) = v5;
      ++*(_DWORD *)(a1 + 16);
    }
    while ( v3 != v4 );
LABEL_8:
    v8 = *(_QWORD **)(a2 + 32);
    for ( i = *(_QWORD **)(a2 + 40); i != v8; ++*(_DWORD *)(a1 + 16) )
    {
      while ( 1 )
      {
        v10 = *v8;
        result = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v8 + 16LL))(*v8);
        if ( (_BYTE)result )
          break;
        if ( i == ++v8 )
          return result;
      }
      result = *(unsigned int *)(a1 + 16);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 20) )
      {
        sub_16CD150(a1 + 8, v13, 0, 8, v11, v12);
        result = *(unsigned int *)(a1 + 16);
      }
      ++v8;
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * result) = v10;
    }
  }
  return result;
}
