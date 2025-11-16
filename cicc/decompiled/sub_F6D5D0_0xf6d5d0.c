// Function: sub_F6D5D0
// Address: 0xf6d5d0
//
__int64 __fastcall sub_F6D5D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h]
  __int64 v17; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v15 = *(_QWORD *)(a2 + 40);
  v16 = *(_QWORD *)(a2 + 32);
  if ( v16 != v15 )
  {
    v7 = a2 + 56;
    do
    {
      v8 = *(_QWORD *)(*(_QWORD *)v16 + 56LL);
      v17 = *(_QWORD *)v16 + 48LL;
      if ( v17 == v8 )
        goto LABEL_13;
      do
      {
        while ( 1 )
        {
          if ( !v8 )
            BUG();
          v9 = *(_QWORD *)(v8 - 8);
          if ( v9 )
            break;
LABEL_12:
          v8 = *(_QWORD *)(v8 + 8);
          if ( v17 == v8 )
            goto LABEL_13;
        }
        while ( 1 )
        {
          v10 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 40LL);
          if ( !*(_BYTE *)(a2 + 84) )
            break;
          v11 = *(_QWORD **)(a2 + 64);
          v12 = &v11[*(unsigned int *)(a2 + 76)];
          if ( v11 == v12 )
            goto LABEL_16;
          while ( v10 != *v11 )
          {
            if ( v12 == ++v11 )
              goto LABEL_16;
          }
LABEL_11:
          v9 = *(_QWORD *)(v9 + 8);
          if ( !v9 )
            goto LABEL_12;
        }
        if ( sub_C8CA60(v7, v10) )
          goto LABEL_11;
LABEL_16:
        v14 = *(unsigned int *)(a1 + 8);
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v14 + 1, 8u, a5, a6);
          v14 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v14) = v8 - 24;
        ++*(_DWORD *)(a1 + 8);
        v8 = *(_QWORD *)(v8 + 8);
      }
      while ( v17 != v8 );
LABEL_13:
      v16 += 8;
    }
    while ( v15 != v16 );
  }
  return a1;
}
