// Function: sub_31F1F80
// Address: 0x31f1f80
//
__int64 __fastcall sub_31F1F80(__int64 a1, const void *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  _QWORD *v6; // r13
  _QWORD *v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rdi
  _QWORD *v12; // rdi
  unsigned __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v19; // [rsp+18h] [rbp-68h] BYREF
  __int64 v20[4]; // [rsp+20h] [rbp-60h] BYREF
  char v21; // [rsp+40h] [rbp-40h]
  char v22; // [rsp+41h] [rbp-3Fh]

  v5 = *(_QWORD *)(a1 + 240);
  v6 = *(_QWORD **)(v5 + 2480);
  v7 = (_QWORD *)(v5 + 8);
  if ( !v6 )
    v6 = v7;
  sub_E65170((__int64)v6);
  v8 = (__int64 *)v6[11];
  v22 = 1;
  v20[0] = (__int64)"<inline asm>";
  v21 = 3;
  sub_C7DE20(&v19, a2, a3, (__int64)v20, v9, v10);
  v20[1] = 0;
  v20[2] = 0;
  v20[0] = v19;
  v11 = (_QWORD *)v8[1];
  if ( v11 == (_QWORD *)v8[2] )
  {
    sub_C12520(v8, v8[1], (__int64)v20);
    v12 = (_QWORD *)v8[1];
  }
  else
  {
    if ( v11 )
    {
      sub_C8EDF0(v11, v20);
      v11 = (_QWORD *)v8[1];
    }
    v12 = v11 + 3;
    v8[1] = (__int64)v12;
  }
  v13 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v12 - *v8) >> 3);
  sub_C8EE20(v20);
  if ( a4 )
  {
    v14 = v6[12];
    v15 = (v6[13] - v14) >> 3;
    if ( (unsigned int)v13 > v15 )
    {
      sub_31F1DD0((__int64)(v6 + 12), (unsigned int)v13 - v15);
      *(_QWORD *)(v6[12] + 8LL * (unsigned int)(v13 - 1)) = a4;
    }
    else
    {
      if ( (unsigned int)v13 < v15 )
      {
        v16 = v14 + 8LL * (unsigned int)v13;
        if ( v6[13] != v16 )
          v6[13] = v16;
      }
      *(_QWORD *)(v14 + 8LL * (unsigned int)(v13 - 1)) = a4;
    }
  }
  return (unsigned int)v13;
}
