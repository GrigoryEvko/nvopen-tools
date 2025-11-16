// Function: sub_37E1680
// Address: 0x37e1680
//
__int64 __fastcall sub_37E1680(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 (__fastcall *a6)(__int64, __int64))
{
  __int64 v6; // r14
  __int64 i; // r15
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 result; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v19; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+30h] [rbp-40h] BYREF
  __int64 v22; // [rsp+38h] [rbp-38h]

  v6 = (a3 - 1) / 2;
  v19 = a3 & 1;
  if ( a2 >= v6 )
  {
    v9 = a1 + 16 * a2;
    if ( (a3 & 1) != 0 )
    {
      v21 = a4;
      v22 = a5;
      goto LABEL_13;
    }
    v8 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    v9 = a1 + 32 * (i + 1);
    if ( a6(v9, v9 - 16) )
    {
      --v8;
      v9 = a1 + 16 * v8;
    }
    v10 = a1 + 16 * i;
    *(_QWORD *)v10 = *(_QWORD *)v9;
    *(_DWORD *)(v10 + 8) = *(_DWORD *)(v9 + 8);
    if ( v8 >= v6 )
      break;
  }
  if ( !v19 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v8 )
    {
      v14 = v8 + 1;
      v8 = 2 * (v8 + 1) - 1;
      v15 = a1 + 32 * v14 - 16;
      *(_QWORD *)v9 = *(_QWORD *)v15;
      *(_DWORD *)(v9 + 8) = *(_DWORD *)(v15 + 8);
      v9 = a1 + 16 * v8;
    }
  }
  v21 = a4;
  v22 = a5;
  v11 = (v8 - 1) / 2;
  if ( v8 > a2 )
  {
    while ( 1 )
    {
      v12 = a1 + 16 * v11;
      v9 = a1 + 16 * v8;
      if ( !a6(v12, (__int64)&v21) )
        break;
      v8 = v11;
      *(_QWORD *)v9 = *(_QWORD *)v12;
      *(_DWORD *)(v9 + 8) = *(_DWORD *)(v12 + 8);
      if ( a2 >= v11 )
      {
        v9 = a1 + 16 * v11;
        break;
      }
      v11 = (v11 - 1) / 2;
    }
  }
LABEL_13:
  *(_QWORD *)v9 = v21;
  result = (unsigned int)v22;
  *(_DWORD *)(v9 + 8) = v22;
  return result;
}
