// Function: sub_1399010
// Address: 0x1399010
//
__int64 __fastcall sub_1399010(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 result; // rax
  unsigned __int64 v7; // rbx
  _QWORD *v8; // r14
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int64 *v13; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 2;
  v3 = (_QWORD *)a1[3];
  v12 = a2;
  if ( !v3 )
    goto LABEL_10;
  do
  {
    while ( 1 )
    {
      v4 = v3[2];
      v5 = v3[3];
      if ( v3[4] >= a2 )
        break;
      v3 = (_QWORD *)v3[3];
      if ( !v5 )
        goto LABEL_6;
    }
    v2 = v3;
    v3 = (_QWORD *)v3[2];
  }
  while ( v4 );
LABEL_6:
  if ( a1 + 2 == v2 || v2[4] > a2 )
  {
LABEL_10:
    v13 = &v12;
    v2 = (_QWORD *)sub_1398F60(a1 + 1, v2, &v13);
    result = v2[5];
    if ( result )
      return result;
  }
  else
  {
    result = v2[5];
    if ( result )
      return result;
  }
  v7 = v12;
  result = sub_22077B0(40);
  if ( result )
  {
    *(_QWORD *)result = v7;
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_DWORD *)(result + 32) = 0;
  }
  v8 = (_QWORD *)v2[5];
  v2[5] = result;
  if ( v8 )
  {
    v9 = v8[2];
    v10 = v8[1];
    if ( v9 != v10 )
    {
      do
      {
        v11 = *(_QWORD *)(v10 + 16);
        if ( v11 != 0 && v11 != -8 && v11 != -16 )
          sub_1649B30(v10);
        v10 += 32;
      }
      while ( v9 != v10 );
      v10 = v8[1];
    }
    if ( v10 )
      j_j___libc_free_0(v10, v8[3] - v10);
    j_j___libc_free_0(v8, 40);
    return v2[5];
  }
  return result;
}
