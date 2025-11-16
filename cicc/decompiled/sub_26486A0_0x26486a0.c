// Function: sub_26486A0
// Address: 0x26486a0
//
__int64 *__fastcall sub_26486A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 *v5; // r14
  __int64 v6; // rbx
  char *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 v12; // rsi
  __int64 v14; // [rsp+8h] [rbp-88h]
  char *v15; // [rsp+10h] [rbp-80h]
  unsigned int v16; // [rsp+1Ch] [rbp-74h]
  char v17; // [rsp+20h] [rbp-70h] BYREF
  unsigned int *v18; // [rsp+30h] [rbp-60h]
  _QWORD v19[10]; // [rsp+40h] [rbp-50h] BYREF

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 4;
  if ( v4 > 0 )
  {
    v8 = &v17;
    do
    {
      while ( 1 )
      {
        v11 = &v5[2 * (v6 >> 1)];
        v12 = *v11;
        if ( !*(_DWORD *)(*v11 + 40) )
          goto LABEL_8;
        if ( *(_DWORD *)(*(_QWORD *)a3 + 40LL) )
          break;
LABEL_6:
        v5 = v11 + 2;
        v6 = v6 - (v6 >> 1) - 1;
        if ( v6 <= 0 )
          return v5;
      }
      v9 = *(unsigned __int8 *)(v12 + 16);
      v10 = *(unsigned __int8 *)(*(_QWORD *)a3 + 16LL);
      if ( (_BYTE)v9 == (_BYTE)v10 )
      {
        v14 = a4;
        v15 = v8;
        sub_22B0690(v8, (__int64 *)(v12 + 24));
        v16 = *v18;
        sub_22B0690(v19, (__int64 *)(*(_QWORD *)a3 + 24LL));
        v8 = v15;
        a4 = v14;
        if ( v16 >= *(_DWORD *)v19[2] )
        {
          v6 >>= 1;
          continue;
        }
        goto LABEL_6;
      }
      if ( *(_DWORD *)(a4 + 4 * v9) < *(_DWORD *)(a4 + 4 * v10) )
        goto LABEL_6;
LABEL_8:
      v6 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v5;
}
