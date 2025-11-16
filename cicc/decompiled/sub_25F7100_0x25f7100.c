// Function: sub_25F7100
// Address: 0x25f7100
//
_QWORD *__fastcall sub_25F7100(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  _QWORD *v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 *v7; // r13
  __int64 v8; // r14
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // rax
  int v12; // eax
  unsigned int v13; // edx
  int v14; // eax
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 v17; // [rsp+10h] [rbp-40h]
  unsigned int v18; // [rsp+18h] [rbp-38h]
  unsigned int v19; // [rsp+1Ch] [rbp-34h]

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    v16 = *a3;
    v15 = *a3 + 24;
    v19 = *(_DWORD *)(*a3 + 32);
    do
    {
      while ( 1 )
      {
        v6 = v5 >> 1;
        v7 = &v4[v5 >> 1];
        v8 = *v7;
        if ( v19 > 0x40 )
        {
          v14 = sub_C444A0(v15);
          v9 = -1;
          if ( v19 - v14 <= 0x40 )
            v9 = **(_QWORD **)(v16 + 24);
        }
        else
        {
          v9 = *(_QWORD *)(v16 + 24);
        }
        v18 = *(_DWORD *)(v8 + 32);
        if ( v18 > 0x40 )
        {
          v17 = v9;
          v12 = sub_C444A0(v8 + 24);
          v9 = v17;
          v13 = v18 - v12;
          v10 = -1;
          if ( v13 <= 0x40 )
            v10 = **(_QWORD **)(v8 + 24);
        }
        else
        {
          v10 = *(_QWORD *)(v8 + 24);
        }
        if ( v10 > v9 )
          break;
        v4 = v7 + 1;
        v5 = v5 - v6 - 1;
        if ( v5 <= 0 )
          return v4;
      }
      v5 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v4;
}
