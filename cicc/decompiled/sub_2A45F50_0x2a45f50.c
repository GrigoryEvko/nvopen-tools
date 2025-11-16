// Function: sub_2A45F50
// Address: 0x2a45f50
//
void __fastcall sub_2A45F50(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  int v13; // ebx
  _QWORD *v14; // [rsp+8h] [rbp-48h]
  unsigned __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x18u, v15, a6);
  v7 = *(_QWORD **)a1;
  v8 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v8 )
  {
    v9 = (unsigned __int64 *)v6;
    do
    {
      if ( v9 )
      {
        *v9 = 0;
        v9[1] = 0;
        v10 = v7[2];
        v9[2] = v10;
        if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        {
          v14 = v7;
          sub_BD6050(v9, *v7 & 0xFFFFFFFFFFFFFFF8LL);
          v7 = v14;
        }
      }
      v7 += 3;
      v9 += 3;
    }
    while ( v8 != v7 );
    v11 = *(_QWORD **)a1;
    v8 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v8 )
    {
      do
      {
        v12 = *(v8 - 1);
        v8 -= 3;
        if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
          sub_BD60C0(v8);
      }
      while ( v8 != v11 );
      v8 = *(_QWORD **)a1;
    }
  }
  v13 = v15[0];
  if ( (_QWORD *)(a1 + 16) != v8 )
    _libc_free((unsigned __int64)v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v13;
}
