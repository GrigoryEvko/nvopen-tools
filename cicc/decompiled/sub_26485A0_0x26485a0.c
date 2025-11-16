// Function: sub_26485A0
// Address: 0x26485a0
//
__int64 __fastcall sub_26485A0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r14
  __int64 v6; // rbx
  char *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v13; // r12
  __int64 v15; // [rsp+8h] [rbp-88h]
  char *v16; // [rsp+10h] [rbp-80h]
  unsigned int v17; // [rsp+1Ch] [rbp-74h]
  char v18; // [rsp+20h] [rbp-70h] BYREF
  unsigned int *v19; // [rsp+30h] [rbp-60h]
  _QWORD v20[10]; // [rsp+40h] [rbp-50h] BYREF

  v4 = a2 - a1;
  v5 = a1;
  v6 = v4 >> 4;
  if ( v4 > 0 )
  {
    v8 = &v18;
    do
    {
      while ( 1 )
      {
        v11 = *a3;
        v12 = v6 >> 1;
        v13 = v5 + 16 * (v6 >> 1);
        if ( !*(_DWORD *)(*a3 + 40) )
          goto LABEL_8;
        if ( *(_DWORD *)(*(_QWORD *)v13 + 40LL) )
          break;
LABEL_6:
        v6 >>= 1;
        if ( v12 <= 0 )
          return v5;
      }
      v9 = *(unsigned __int8 *)(v11 + 16);
      v10 = *(unsigned __int8 *)(*(_QWORD *)v13 + 16LL);
      if ( (_BYTE)v9 == (_BYTE)v10 )
      {
        v15 = a4;
        v16 = v8;
        sub_22B0690(v8, (__int64 *)(v11 + 24));
        v17 = *v19;
        sub_22B0690(v20, (__int64 *)(*(_QWORD *)v13 + 24LL));
        v8 = v16;
        a4 = v15;
        if ( v17 >= *(_DWORD *)v20[2] )
        {
          v5 = v13 + 16;
          v6 = v6 - v12 - 1;
          continue;
        }
        goto LABEL_6;
      }
      if ( *(_DWORD *)(a4 + 4 * v9) < *(_DWORD *)(a4 + 4 * v10) )
        goto LABEL_6;
LABEL_8:
      v5 = v13 + 16;
      v6 = v6 - v12 - 1;
    }
    while ( v6 > 0 );
  }
  return v5;
}
