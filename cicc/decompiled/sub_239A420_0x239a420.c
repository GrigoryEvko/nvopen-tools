// Function: sub_239A420
// Address: 0x239a420
//
__int64 __fastcall sub_239A420(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r15
  _QWORD *v8; // rbx
  __int64 v9; // r13
  unsigned __int8 *v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rsi
  int v13; // ebx
  __int64 v15; // [rsp+8h] [rbp-48h]
  unsigned __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v16, a6);
  v15 = v6;
  v7 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    v8 = (_QWORD *)(*(_QWORD *)a1 + 16LL);
    v9 = v6;
    while ( 1 )
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = *((_DWORD *)v8 - 4);
        *(_QWORD *)(v9 + 8) = *(v8 - 1);
        v10 = (unsigned __int8 *)*v8;
        *(_QWORD *)(v9 + 16) = *v8;
        if ( v10 )
        {
          sub_B976B0((__int64)v8, v10, v9 + 16);
          *v8 = 0;
        }
        *(_QWORD *)(v9 + 24) = v8[1];
      }
      v9 += 32;
      if ( (_QWORD *)v7 == v8 + 2 )
        break;
      v8 += 4;
    }
    v11 = *(_QWORD *)a1;
    v7 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v12 = *(_QWORD *)(v7 - 16);
        v7 -= 32LL;
        if ( v12 )
          sub_B91220(v7 + 16, v12);
      }
      while ( v7 != v11 );
      v7 = *(_QWORD *)a1;
    }
  }
  v13 = v16[0];
  if ( a1 + 16 != v7 )
    _libc_free(v7);
  *(_DWORD *)(a1 + 12) = v13;
  *(_QWORD *)a1 = v15;
  return v15;
}
