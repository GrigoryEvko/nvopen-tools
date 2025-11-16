// Function: sub_1623260
// Address: 0x1623260
//
__int64 __fastcall sub_1623260(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  __int64 v8; // r15
  _QWORD *i; // rbx
  unsigned __int8 *v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v3 = ((((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
      | (a1[3] + 2LL)
      | (((unsigned __int64)a1[3] + 2) >> 1)) >> 4;
  v4 = ((v3
       | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
       | (a1[3] + 2LL)
       | (((unsigned __int64)a1[3] + 2) >> 1)) >> 8)
     | v3
     | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
     | (a1[3] + 2LL)
     | (((unsigned __int64)a1[3] + 2) >> 1);
  v5 = (v4 | (v4 >> 16) | HIDWORD(v4)) + 1;
  if ( v5 >= a2 )
    v2 = v5;
  v6 = v2;
  if ( v2 > 0xFFFFFFFF )
    v6 = 0xFFFFFFFFLL;
  v14 = malloc(16 * v6);
  if ( !v14 )
    sub_16BD1C0("Allocation failed");
  v7 = *(_QWORD *)a1 + 16LL * a1[2];
  if ( *(_QWORD *)a1 != v7 )
  {
    v8 = v14;
    for ( i = (_QWORD *)(*(_QWORD *)a1 + 8LL); ; i += 2 )
    {
      if ( v8 )
      {
        *(_DWORD *)v8 = *((_DWORD *)i - 2);
        v10 = (unsigned __int8 *)*i;
        *(_QWORD *)(v8 + 8) = *i;
        if ( v10 )
        {
          sub_1623210((__int64)i, v10, v8 + 8);
          *i = 0;
        }
      }
      v8 += 16;
      if ( (_QWORD *)v7 == i + 1 )
        break;
    }
    v11 = *(_QWORD *)a1;
    v7 = *(_QWORD *)a1 + 16LL * a1[2];
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v12 = *(_QWORD *)(v7 - 8);
        v7 -= 16LL;
        if ( v12 )
          sub_161E7C0(v7 + 8, v12);
      }
      while ( v7 != v11 );
      v7 = *(_QWORD *)a1;
    }
  }
  if ( (unsigned int *)v7 != a1 + 4 )
    _libc_free(v7);
  a1[3] = v6;
  *(_QWORD *)a1 = v14;
  return v14;
}
