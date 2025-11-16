// Function: sub_29F41F0
// Address: 0x29f41f0
//
__int64 __fastcall sub_29F41F0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r14
  const char *v3; // rax
  size_t v4; // rdx
  size_t v5; // r13
  const char *v6; // r15
  const char *v7; // rax
  size_t v8; // rdx
  size_t v9; // r12
  int v10; // eax
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v1 = a1;
  v2 = a1[1];
  v13 = *a1;
  while ( 1 )
  {
    v3 = sub_BD5D20(*(v1 - 1));
    v5 = v4;
    v6 = v3;
    v7 = sub_BD5D20(v2);
    v9 = v8;
    if ( v5 <= v8 )
      v8 = v5;
    if ( !v8 )
      break;
    v10 = memcmp(v7, v6, v8);
    if ( !v10 )
      break;
    if ( v10 >= 0 )
      goto LABEL_8;
LABEL_10:
    v12 = *(v1 - 2);
    v1 -= 2;
    v1[2] = v12;
    v1[3] = v1[1];
  }
  if ( v5 != v9 && v5 > v9 )
    goto LABEL_10;
LABEL_8:
  v1[1] = v2;
  *v1 = v13;
  return v13;
}
