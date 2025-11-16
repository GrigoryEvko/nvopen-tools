// Function: sub_2308590
// Address: 0x2308590
//
__int64 *__fastcall sub_2308590(__int64 *a1, __int64 a2)
{
  unsigned int v3; // ebx
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r8
  __int64 v9; // rsi
  _QWORD *v10; // rbx
  _QWORD *v11; // r12
  unsigned __int64 v12; // rdi
  _QWORD *v14; // r14
  _QWORD *v15; // rbx
  unsigned __int64 v16; // rdi
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h] BYREF
  _QWORD *v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h]
  unsigned int v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]

  sub_2F7B180(&v19, a2 + 8);
  v3 = v22;
  v4 = (__int64)v20;
  v5 = v23;
  v20 = 0;
  ++v19;
  v17 = v21;
  v21 = 0;
  v22 = 0;
  v6 = sub_22077B0(0x30u);
  v7 = v6;
  if ( v6 )
  {
    *(_DWORD *)(v6 + 32) = v3;
    v8 = 0;
    *(_QWORD *)(v6 + 8) = 1;
    *(_QWORD *)(v6 + 24) = v17;
    *(_QWORD *)v6 = &unk_4A0B290;
    *(_QWORD *)(v6 + 40) = v5;
    *(_QWORD *)(v6 + 16) = v4;
    v4 = 0;
  }
  else
  {
    v8 = 32LL * v3;
    if ( v3 )
    {
      v14 = (_QWORD *)(v4 + v8);
      v15 = (_QWORD *)v4;
      do
      {
        if ( *v15 != -8192 && *v15 != -4096 )
        {
          v16 = v15[1];
          if ( v16 )
          {
            v18 = v8;
            j_j___libc_free_0(v16);
            v8 = v18;
          }
        }
        v15 += 4;
      }
      while ( v14 != v15 );
    }
  }
  sub_C7D6A0(v4, v8, 8);
  v9 = v22;
  *a1 = v7;
  if ( (_DWORD)v9 )
  {
    v10 = v20;
    v11 = &v20[4 * v9];
    do
    {
      if ( *v10 != -8192 && *v10 != -4096 )
      {
        v12 = v10[1];
        if ( v12 )
          j_j___libc_free_0(v12);
      }
      v10 += 4;
    }
    while ( v11 != v10 );
    v9 = v22;
  }
  sub_C7D6A0((__int64)v20, 32 * v9, 8);
  return a1;
}
