// Function: sub_2F32700
// Address: 0x2f32700
//
__int64 __fastcall sub_2F32700(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  unsigned __int64 v7; // r15
  _QWORD *i; // rbx
  unsigned __int8 *v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rsi
  int v12; // ebx
  __int64 v14; // [rsp+8h] [rbp-48h]
  unsigned __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v14 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v15, a6);
  v6 = v14;
  v7 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    for ( i = (_QWORD *)(*(_QWORD *)a1 + 32LL); ; i += 5 )
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = *(i - 4);
        *(_QWORD *)(v6 + 8) = *(i - 3);
        *(_QWORD *)(v6 + 16) = *(i - 2);
        *(_BYTE *)(v6 + 24) = *((_BYTE *)i - 8);
        v9 = (unsigned __int8 *)*i;
        *(_QWORD *)(v6 + 32) = *i;
        if ( v9 )
        {
          sub_B976B0((__int64)i, v9, v6 + 32);
          *i = 0;
        }
      }
      v6 += 40;
      if ( (_QWORD *)v7 == i + 1 )
        break;
    }
    v10 = *(_QWORD *)a1;
    v7 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v11 = *(_QWORD *)(v7 - 8);
        v7 -= 40LL;
        if ( v11 )
          sub_B91220(v7 + 32, v11);
      }
      while ( v7 != v10 );
      v7 = *(_QWORD *)a1;
    }
  }
  v12 = v15[0];
  if ( a1 + 16 != v7 )
    _libc_free(v7);
  *(_DWORD *)(a1 + 12) = v12;
  *(_QWORD *)a1 = v14;
  return v14;
}
