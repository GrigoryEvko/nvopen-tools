// Function: sub_14EA4D0
// Address: 0x14ea4d0
//
__int64 __fastcall sub_14EA4D0(__int64 *a1, int a2, _QWORD *a3)
{
  _QWORD *v3; // r10
  __int64 v4; // r9
  __int64 *v5; // r12
  _QWORD *v7; // rax
  __int64 v8; // r13
  _QWORD *v9; // rax
  __int64 v10; // r15
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdx
  _QWORD *v14; // [rsp-48h] [rbp-48h]
  __int64 v15; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return 0;
  v3 = a3 + 2;
  v4 = (__int64)&a1[(unsigned int)(a2 - 1) + 1];
  v5 = a1;
  do
  {
    v8 = a3[1];
    v9 = (_QWORD *)*a3;
    v10 = *v5;
    v11 = v8 + 1;
    if ( (_QWORD *)*a3 == v3 )
      v12 = 15;
    else
      v12 = a3[2];
    if ( v11 > v12 )
    {
      v14 = v3;
      v15 = v4;
      sub_2240BB0(a3, a3[1], 0, 0, 1);
      v9 = (_QWORD *)*a3;
      v3 = v14;
      v4 = v15;
    }
    *((_BYTE *)v9 + v8) = v10;
    v7 = (_QWORD *)*a3;
    ++v5;
    a3[1] = v11;
    *((_BYTE *)v7 + v8 + 1) = 0;
  }
  while ( v5 != (__int64 *)v4 );
  return 0;
}
