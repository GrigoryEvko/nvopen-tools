// Function: sub_1517EB0
// Address: 0x1517eb0
//
__int64 __fastcall sub_1517EB0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r12
  unsigned __int64 v7; // r13
  unsigned int v8; // r14d
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // r13
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // rsi
  unsigned int v15; // [rsp+Ch] [rbp-54h] BYREF
  _BYTE v16[80]; // [rsp+10h] [rbp-50h] BYREF

  v2 = a2;
  v3 = *(unsigned int *)(a1 + 8);
  v15 = a2;
  if ( a2 < (unsigned int)v3 )
    goto LABEL_2;
  v7 = a2 + 1;
  v8 = a2 + 1;
  if ( v7 < v3 )
  {
    v4 = *(_QWORD *)a1;
    v12 = *(_QWORD *)a1 + 8 * v3;
    v13 = *(_QWORD *)a1 + 8 * v7;
    if ( v12 != v13 )
    {
      do
      {
        v14 = *(_QWORD *)(v12 - 8);
        v12 -= 8;
        if ( v14 )
          sub_161E7C0(v12);
      }
      while ( v13 != v12 );
      v2 = v15;
      v4 = *(_QWORD *)a1;
    }
    *(_DWORD *)(a1 + 8) = v8;
  }
  else
  {
    if ( v7 <= v3 )
    {
LABEL_2:
      v4 = *(_QWORD *)a1;
      goto LABEL_3;
    }
    if ( v7 > *(unsigned int *)(a1 + 12) )
    {
      sub_1516630(a1, v7);
      v3 = *(unsigned int *)(a1 + 8);
    }
    v4 = *(_QWORD *)a1;
    v9 = (_QWORD *)(*(_QWORD *)a1 + 8 * v3);
    v10 = *(_QWORD *)a1 + 8 * v7;
    if ( v9 != (_QWORD *)v10 )
    {
      do
      {
        if ( v9 )
          *v9 = 0;
        ++v9;
      }
      while ( (_QWORD *)v10 != v9 );
      v4 = *(_QWORD *)a1;
    }
    *(_DWORD *)(a1 + 8) = v7;
    v2 = v15;
  }
LABEL_3:
  v5 = *(_QWORD *)(v4 + 8 * v2);
  if ( v5 )
    return v5;
  sub_1517B60((__int64)v16, a1 + 24, (int *)&v15);
  v5 = sub_1627350(*(_QWORD *)(a1 + 216), 0, 0, 2, 1);
  v11 = (__int64 *)(*(_QWORD *)a1 + 8LL * v15);
  if ( *v11 )
    sub_161E7C0(*(_QWORD *)a1 + 8LL * v15);
  *v11 = v5;
  if ( !v5 )
    return v5;
  sub_1623A60(v11, v5, 2);
  return v5;
}
