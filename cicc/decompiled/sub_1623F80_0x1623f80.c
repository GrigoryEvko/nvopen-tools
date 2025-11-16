// Function: sub_1623F80
// Address: 0x1623f80
//
__int64 __fastcall sub_1623F80(unsigned int *a1, int a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  _DWORD *v7; // rdi
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 *v10; // rdi

  v5 = *(_QWORD *)a1;
  v6 = a1[2];
  v7 = (_DWORD *)(*(_QWORD *)a1 + 16 * v6);
  if ( v7 == (_DWORD *)v5 )
  {
LABEL_8:
    if ( (unsigned int)v6 >= a1[3] )
    {
      sub_1623400(a1, 0);
      v6 = a1[2];
      v7 = (_DWORD *)(*(_QWORD *)a1 + 16 * v6);
    }
    if ( v7 )
    {
      *v7 = a2;
      v10 = (__int64 *)(v7 + 2);
      *v10 = a3;
      sub_1623A60((__int64)v10, a3, 2);
      LODWORD(v6) = a1[2];
    }
    result = (unsigned int)(v6 + 1);
    a1[2] = result;
  }
  else
  {
    while ( *(_DWORD *)v5 != a2 )
    {
      v5 += 16;
      if ( v7 == (_DWORD *)v5 )
        goto LABEL_8;
    }
    v8 = *(_QWORD *)(v5 + 8);
    if ( v8 )
      sub_161E7C0(v5 + 8, v8);
    *(_QWORD *)(v5 + 8) = a3;
    return sub_1623A60(v5 + 8, a3, 2);
  }
  return result;
}
