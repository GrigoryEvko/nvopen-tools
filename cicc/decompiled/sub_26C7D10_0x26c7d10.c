// Function: sub_26C7D10
// Address: 0x26c7d10
//
void __fastcall sub_26C7D10(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rcx
  _QWORD *v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD **v8; // rax
  _QWORD *v9; // r13
  __int64 v10; // r12
  _QWORD *v11; // r15
  __int64 v12; // r13
  __int64 j; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rdi
  __int64 v18; // rdx
  _QWORD *i; // [rsp+8h] [rbp-98h]
  _QWORD *v20; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v21[2]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 v22; // [rsp+30h] [rbp-70h]
  __int64 v23; // [rsp+38h] [rbp-68h]
  __int64 v24; // [rsp+40h] [rbp-60h]
  unsigned __int64 *v25; // [rsp+48h] [rbp-58h]
  _QWORD *v26; // [rsp+50h] [rbp-50h]
  _QWORD *v27; // [rsp+58h] [rbp-48h]
  __int64 v28; // [rsp+60h] [rbp-40h]
  unsigned __int64 v29; // [rsp+68h] [rbp-38h]

  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v21[1] = 8;
  v21[0] = sub_22077B0(0x40u);
  v2 = v21[0] + 24;
  v3 = sub_22077B0(0x200u);
  v4 = *a1;
  v25 = (unsigned __int64 *)(v21[0] + 24);
  v5 = (_QWORD *)v3;
  *(_QWORD *)(v21[0] + 24) = v3;
  v23 = v3;
  v6 = v3 + 512;
  v24 = v6;
  v29 = v2;
  v27 = v5;
  v28 = v6;
  v26 = v5;
  v7 = *(_QWORD **)(v4 + 24);
  v22 = (unsigned __int64)v5;
  if ( !v7 )
    goto LABEL_22;
  while ( 1 )
  {
    v20 = v7 + 2;
    if ( v5 != (_QWORD *)(v6 - 8) )
      break;
    sub_26C7B10(v21, &v20);
    v7 = (_QWORD *)*v7;
    v5 = v26;
    if ( !v7 )
      goto LABEL_9;
LABEL_6:
    v6 = v28;
  }
  if ( v5 )
  {
    *v5 = v7 + 2;
    v5 = v26;
  }
  v26 = ++v5;
  v7 = (_QWORD *)*v7;
  if ( v7 )
    goto LABEL_6;
LABEL_9:
  v8 = (_QWORD **)v22;
  if ( v5 != (_QWORD *)v22 )
  {
    do
    {
      v9 = *v8;
      if ( v8 == (_QWORD **)(v24 - 8) )
      {
        j_j___libc_free_0(v23);
        v18 = *++v25 + 512;
        v23 = *v25;
        v24 = v18;
        v22 = v23;
      }
      else
      {
        v22 = (unsigned __int64)(v8 + 1);
      }
      v10 = v9[18];
      *v9 = a2;
      v11 = v26;
      for ( i = v9 + 16; i != (_QWORD *)v10; v10 = sub_220EF30(v10) )
      {
        v12 = *(_QWORD *)(v10 + 64);
        for ( j = v10 + 48; j != v12; v12 = sub_220EF30(v12) )
        {
          while ( 1 )
          {
            v20 = (_QWORD *)(v12 + 48);
            if ( v11 != (_QWORD *)(v28 - 8) )
              break;
            sub_26C7B10(v21, &v20);
            v11 = v26;
            v12 = sub_220EF30(v12);
            if ( j == v12 )
              goto LABEL_20;
          }
          if ( v11 )
          {
            *v11 = v12 + 48;
            v11 = v26;
          }
          v26 = ++v11;
        }
LABEL_20:
        ;
      }
      v8 = (_QWORD **)v22;
    }
    while ( (_QWORD *)v22 != v11 );
  }
LABEL_22:
  v14 = v21[0];
  if ( v21[0] )
  {
    v15 = v25;
    v16 = v29 + 8;
    if ( v29 + 8 > (unsigned __int64)v25 )
    {
      do
      {
        v17 = *v15++;
        j_j___libc_free_0(v17);
      }
      while ( v16 > (unsigned __int64)v15 );
      v14 = v21[0];
    }
    j_j___libc_free_0(v14);
  }
}
