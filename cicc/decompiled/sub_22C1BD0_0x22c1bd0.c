// Function: sub_22C1BD0
// Address: 0x22c1bd0
//
void __fastcall sub_22C1BD0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // r13
  _QWORD *v4; // rbx
  _QWORD *i; // r13
  __int64 v6; // rax
  char v7; // al
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // r13
  __int64 j; // rax
  __int64 v12; // rdx
  __int64 v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]
  __int64 v18; // [rsp+30h] [rbp-30h]

  v1 = *a1;
  if ( !*a1 )
    return;
  if ( *(_BYTE *)(v1 + 448) )
  {
    *(_BYTE *)(v1 + 448) = 0;
    sub_22C1AA0(v1 + 384);
    if ( (*(_BYTE *)(v1 + 392) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v1 + 400), 24LL * *(unsigned int *)(v1 + 408), 8);
  }
  if ( (*(_BYTE *)(v1 + 280) & 1) != 0 )
  {
    v13 = 0;
    v4 = (_QWORD *)(v1 + 288);
    v3 = 12;
    v14 = 0;
    v15 = -4096;
    v16 = 0;
    v17 = 0;
    v18 = -8192;
  }
  else
  {
    v2 = *(unsigned int *)(v1 + 296);
    v3 = 3 * v2;
    if ( !(_DWORD)v2 )
      goto LABEL_34;
    v4 = *(_QWORD **)(v1 + 288);
    v13 = 0;
    v14 = 0;
    v15 = -4096;
    v16 = 0;
    v17 = 0;
    v18 = -8192;
  }
  for ( i = &v4[v3]; i != v4; v4 += 3 )
  {
    v6 = v4[2];
    if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
      sub_BD60C0(v4);
  }
  sub_D68D70(&v16);
  sub_D68D70(&v13);
  if ( (*(_BYTE *)(v1 + 280) & 1) != 0 )
    goto LABEL_12;
  v2 = *(unsigned int *)(v1 + 296);
LABEL_34:
  sub_C7D6A0(*(_QWORD *)(v1 + 288), 24 * v2, 8);
LABEL_12:
  v7 = *(_BYTE *)(v1 + 8);
  if ( (v7 & 1) != 0 || *(_DWORD *)(v1 + 24) )
  {
    v13 = 0;
    v14 = 0;
    v15 = -4096;
    v16 = 0;
    v17 = 0;
    v18 = -8192;
    if ( (*(_BYTE *)(v1 + 8) & 1) != 0 )
    {
      v8 = v1 + 16;
      v9 = 256;
    }
    else
    {
      v8 = *(_QWORD *)(v1 + 16);
      v9 = (unsigned __int64)*(unsigned int *)(v1 + 24) << 6;
    }
    v10 = v8 + v9;
    if ( v10 != v8 )
    {
      for ( j = -4096; ; j = v15 )
      {
        v12 = *(_QWORD *)(v8 + 16);
        if ( v12 != j )
        {
          j = v18;
          if ( v12 != v18 )
          {
            sub_22C0090((unsigned __int8 *)(v8 + 24));
            j = *(_QWORD *)(v8 + 16);
          }
        }
        if ( j != 0 && j != -4096 && j != -8192 )
          sub_BD60C0((_QWORD *)v8);
        v8 += 64LL;
        if ( v10 == v8 )
          break;
      }
    }
    sub_D68D70(&v16);
    sub_D68D70(&v13);
    v7 = *(_BYTE *)(v1 + 8);
  }
  if ( (v7 & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(v1 + 16), (unsigned __int64)*(unsigned int *)(v1 + 24) << 6, 8);
  j_j___libc_free_0(v1);
}
