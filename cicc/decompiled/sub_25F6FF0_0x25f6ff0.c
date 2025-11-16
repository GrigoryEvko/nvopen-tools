// Function: sub_25F6FF0
// Address: 0x25f6ff0
//
__int64 *__fastcall sub_25F6FF0(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 *v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 *v8; // r13
  __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  unsigned int v13; // edx
  __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-48h]
  int v16; // [rsp+10h] [rbp-40h]
  unsigned int v17; // [rsp+1Ch] [rbp-34h]

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    v6 = *a3;
    v14 = *a3 + 24;
    v17 = *(_DWORD *)(*a3 + 32);
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = &v4[v5 >> 1];
        v9 = *v8;
        if ( *(_DWORD *)(*v8 + 32) > 0x40u )
        {
          v15 = *v8;
          v10 = -1;
          v16 = *(_DWORD *)(*v8 + 32);
          if ( v16 - (unsigned int)sub_C444A0(v9 + 24) <= 0x40 )
            v10 = **(_QWORD **)(v15 + 24);
        }
        else
        {
          v10 = *(_QWORD *)(v9 + 24);
        }
        if ( v17 > 0x40 )
        {
          v13 = v17 - sub_C444A0(v14);
          v11 = -1;
          if ( v13 <= 0x40 )
            v11 = **(_QWORD **)(v6 + 24);
        }
        else
        {
          v11 = *(_QWORD *)(v6 + 24);
        }
        if ( v11 <= v10 )
          break;
        v4 = v8 + 1;
        v5 = v5 - v7 - 1;
        if ( v5 <= 0 )
          return v4;
      }
      v5 >>= 1;
    }
    while ( v7 > 0 );
  }
  return v4;
}
