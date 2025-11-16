// Function: sub_1E64080
// Address: 0x1e64080
//
__int64 __fastcall sub_1E64080(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12
  __int64 v4; // rcx
  __int64 v5; // rdx
  unsigned __int64 *v6; // rax
  unsigned __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 *v9; // [rsp+8h] [rbp-28h]

  v2 = (_QWORD *)a1[10];
  if ( !v2 )
    goto LABEL_8;
  v3 = a1 + 9;
  do
  {
    while ( 1 )
    {
      v4 = v2[2];
      v5 = v2[3];
      if ( v2[4] >= a2 )
        break;
      v2 = (_QWORD *)v2[3];
      if ( !v5 )
        goto LABEL_6;
    }
    v3 = v2;
    v2 = (_QWORD *)v2[2];
  }
  while ( v4 );
LABEL_6:
  if ( a1 + 9 == v3 || v3[4] > a2 )
  {
LABEL_8:
    v6 = (unsigned __int64 *)sub_22077B0(16);
    if ( v6 )
    {
      v6[1] = (unsigned __int64)a1;
      *v6 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    }
    v8 = a2;
    v9 = v6;
    v3 = sub_1E63F40(a1 + 8, &v8);
    if ( v9 )
      j_j___libc_free_0(v9, 16);
  }
  return v3[5];
}
