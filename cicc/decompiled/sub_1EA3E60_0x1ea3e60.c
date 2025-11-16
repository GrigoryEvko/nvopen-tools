// Function: sub_1EA3E60
// Address: 0x1ea3e60
//
__int64 __fastcall sub_1EA3E60(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  unsigned __int64 v6; // r15
  __int64 v7; // r14
  _QWORD *i; // rbx
  unsigned __int8 *v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = (((((((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
         | (a1[3] + 2LL)
         | (((unsigned __int64)a1[3] + 2) >> 1)) >> 4)
       | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
       | (a1[3] + 2LL)
       | (((unsigned __int64)a1[3] + 2) >> 1)) >> 8)
     | (((((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
       | (a1[3] + 2LL)
       | (((unsigned __int64)a1[3] + 2) >> 1)) >> 4)
     | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
     | (a1[3] + 2LL)
     | (((unsigned __int64)a1[3] + 2) >> 1);
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v4 >= a2 )
    v2 = v4;
  v5 = v2;
  if ( v2 > 0xFFFFFFFF )
    v5 = 0xFFFFFFFFLL;
  v13 = malloc(40 * v5);
  if ( !v13 )
    sub_16BD1C0("Allocation failed", 1u);
  v6 = *(_QWORD *)a1 + 40LL * a1[2];
  if ( *(_QWORD *)a1 != v6 )
  {
    v7 = v13;
    for ( i = (_QWORD *)(*(_QWORD *)a1 + 32LL); ; i += 5 )
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = *(i - 4);
        *(_QWORD *)(v7 + 8) = *(i - 3);
        *(_QWORD *)(v7 + 16) = *(i - 2);
        *(_BYTE *)(v7 + 24) = *((_BYTE *)i - 8);
        v9 = (unsigned __int8 *)*i;
        *(_QWORD *)(v7 + 32) = *i;
        if ( v9 )
        {
          sub_1623210((__int64)i, v9, v7 + 32);
          *i = 0;
        }
      }
      v7 += 40;
      if ( (_QWORD *)v6 == i + 1 )
        break;
    }
    v10 = *(_QWORD *)a1;
    v6 = *(_QWORD *)a1 + 40LL * a1[2];
    if ( *(_QWORD *)a1 != v6 )
    {
      do
      {
        v11 = *(_QWORD *)(v6 - 8);
        v6 -= 40LL;
        if ( v11 )
          sub_161E7C0(v6 + 32, v11);
      }
      while ( v6 != v10 );
      v6 = *(_QWORD *)a1;
    }
  }
  if ( (unsigned int *)v6 != a1 + 4 )
    _libc_free(v6);
  a1[3] = v5;
  *(_QWORD *)a1 = v13;
  return v13;
}
