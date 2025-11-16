// Function: sub_1E78C80
// Address: 0x1e78c80
//
char *__fastcall sub_1E78C80(__int64 *src, __int64 *a2, __int64 *a3, __int64 *a4, _QWORD *a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 *v7; // r12
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rax
  char *v16; // rbx
  unsigned __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  __int64 v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = a3;
  v7 = src;
  v21[0] = a6;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v12 = *v7;
      v13 = *v6;
      v14 = *(_QWORD *)(v21[0] + 280);
      v20 = v21[0];
      if ( v14
        && (v19 = sub_1DDC3C0(v14, v13), (v9 = *(_QWORD *)(v20 + 280)) != 0)
        && (v10 = sub_1DDC3C0(v9, v12), v19)
        && v10 )
      {
        if ( v19 < v10 )
          goto LABEL_8;
      }
      else if ( sub_1E78020((__int64)v21, v13, v12) )
      {
LABEL_8:
        v11 = *v6;
        ++a5;
        ++v6;
        *(a5 - 1) = v11;
        if ( v7 == a2 )
          break;
        continue;
      }
      v15 = *v7;
      ++a5;
      ++v7;
      *(a5 - 1) = v15;
      if ( v7 == a2 )
        break;
    }
    while ( v6 != a4 );
  }
  if ( a2 != v7 )
    memmove(a5, v7, (char *)a2 - (char *)v7);
  v16 = (char *)a5 + (char *)a2 - (char *)v7;
  if ( a4 != v6 )
    memmove(v16, v6, (char *)a4 - (char *)v6);
  return &v16[(char *)a4 - (char *)v6];
}
