// Function: sub_3148160
// Address: 0x3148160
//
__int64 __fastcall sub_3148160(unsigned __int16 *a1, int a2, __int64 a3)
{
  unsigned __int16 *v5; // rbx
  unsigned __int16 *v6; // r12
  __int64 v7; // rdx
  __int16 *v8; // rax
  int v9; // edi
  __int16 *v10; // rax
  int v11; // esi
  __int64 v12; // rsi
  int v14; // [rsp+1Ch] [rbp-64h] BYREF
  int v15; // [rsp+20h] [rbp-60h] BYREF
  __int16 *v16; // [rsp+28h] [rbp-58h]
  __int16 v17; // [rsp+30h] [rbp-50h]
  int v18; // [rsp+38h] [rbp-48h]
  __int64 v19; // [rsp+40h] [rbp-40h]
  __int16 v20; // [rsp+48h] [rbp-38h]

  v5 = &a1[20 * *a1 + 20 + *((unsigned __int8 *)a1 + 8) + (unsigned __int64)*((unsigned int *)a1 + 3)];
  v6 = &v5[*((unsigned __int8 *)a1 + 9)];
  if ( v6 == v5 )
    return 0;
  while ( 1 )
  {
    v12 = *v5;
    if ( (_DWORD)v12 == a2 )
      break;
    if ( a3 )
    {
      v7 = *(_QWORD *)(a3 + 8);
      v14 = a2;
      v18 = 0;
      v19 = 0;
      v8 = (__int16 *)(*(_QWORD *)(a3 + 56) + 2LL * *(unsigned int *)(v7 + 24 * v12 + 8));
      v9 = *v8;
      v10 = v8 + 1;
      v11 = v9 + v12;
      if ( !(_WORD)v9 )
        v10 = 0;
      v15 = v11;
      v17 = v11;
      v16 = v10;
      v20 = 0;
      if ( sub_2E46590(&v15, &v14) )
        break;
    }
    if ( v6 == ++v5 )
      return 0;
  }
  return 1;
}
