// Function: sub_B97B00
// Address: 0xb97b00
//
__int64 __fastcall sub_B97B00(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v3; // rsi
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rbx
  int v9; // ebx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (unsigned __int8 *)(a1 + 2);
  v4 = sub_C8D7D0(a1, a1 + 2, a2, 16, v12);
  v11 = v4;
  v5 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v5 )
  {
    v6 = (_QWORD *)(*a1 + 8);
    v7 = v4;
    while ( 1 )
    {
      if ( v7 )
      {
        *(_DWORD *)v7 = *((_DWORD *)v6 - 2);
        v3 = (unsigned __int8 *)*v6;
        *(_QWORD *)(v7 + 8) = *v6;
        if ( v3 )
        {
          sub_B976B0((__int64)v6, v3, v7 + 8);
          *v6 = 0;
        }
      }
      v7 += 16;
      if ( (_QWORD *)v5 == v6 + 1 )
        break;
      v6 += 2;
    }
    v8 = *a1;
    v5 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v5 )
    {
      do
      {
        v3 = *(unsigned __int8 **)(v5 - 8);
        v5 -= 16;
        if ( v3 )
          sub_B91220(v5 + 8, (__int64)v3);
      }
      while ( v5 != v8 );
      v5 = *a1;
    }
  }
  v9 = v12[0];
  if ( a1 + 2 != (__int64 *)v5 )
    _libc_free(v5, v3);
  *((_DWORD *)a1 + 3) = v9;
  *a1 = v11;
  return v11;
}
