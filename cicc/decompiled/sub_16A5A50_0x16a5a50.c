// Function: sub_16A5A50
// Address: 0x16a5a50
//
__int64 __fastcall sub_16A5A50(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 *v4; // r12
  __int64 v6; // rsi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // ecx

  v4 = a2;
  if ( a3 > 0x40 )
  {
    v8 = sub_2207820(8 * (((unsigned __int64)a3 + 63) >> 6));
    v9 = *a2;
    v10 = v8;
    v11 = 0;
    do
    {
      *(_QWORD *)(v10 + v11) = *(_QWORD *)(v9 + v11);
      v11 += 8;
    }
    while ( v11 != 8LL * (a3 >> 6) );
    v12 = -a3 & 0x3F;
    if ( v12 )
      *(_QWORD *)(v10 + v11) = *(_QWORD *)(v9 + v11) << v12 >> v12;
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = v10;
    return a1;
  }
  else
  {
    if ( *((_DWORD *)a2 + 2) > 0x40u )
      v4 = (__int64 *)*a2;
    v6 = *v4;
    *(_DWORD *)(a1 + 8) = a3;
    *(_QWORD *)a1 = v6 & (0xFFFFFFFFFFFFFFFFLL >> -(char)a3);
    return a1;
  }
}
