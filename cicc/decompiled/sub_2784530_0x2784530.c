// Function: sub_2784530
// Address: 0x2784530
//
__int64 __fastcall sub_2784530(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rax
  bool v7; // zf
  char *v8; // rdi
  __int64 v9; // rax
  char *v10; // r12
  char *v11; // rbx
  unsigned int v12; // r14d
  unsigned int v13; // r15d
  __int64 v14; // rdi
  unsigned int v15; // eax
  char *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-80h]
  char *v20; // [rsp+10h] [rbp-70h] BYREF
  char *v21; // [rsp+18h] [rbp-68h]
  char *i; // [rsp+20h] [rbp-60h]
  _QWORD v23[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+40h] [rbp-40h]

  v2 = a1 + 72;
  v3 = *(_QWORD *)(a1 + 80);
  v20 = 0;
  v21 = 0;
  i = 0;
  if ( a1 + 72 != v3 )
  {
    v4 = 0;
    do
    {
      v3 = *(_QWORD *)(v3 + 8);
      ++v4;
    }
    while ( v3 != v2 );
    if ( v4 > 0x555555555555555LL )
      sub_4262D8((__int64)"vector::reserve");
    v19 = 24 * v4;
    v20 = (char *)sub_22077B0(24 * v4);
    v21 = v20;
    v5 = *(_QWORD *)(a1 + 80);
    for ( i = &v20[v19]; v2 != v5; v5 = *(_QWORD *)(v5 + 8) )
    {
      v9 = v5 - 24;
      v23[0] = 4;
      if ( !v5 )
        v9 = 0;
      v23[1] = 0;
      v24 = v9;
      if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
        sub_BD73F0((__int64)v23);
      v8 = v21;
      if ( v21 == i )
      {
        sub_913E90(&v20, v21, v23);
      }
      else
      {
        if ( v21 )
        {
          *(_QWORD *)v21 = 4;
          *((_QWORD *)v8 + 1) = 0;
          v6 = v24;
          v7 = v24 == 0;
          *((_QWORD *)v8 + 2) = v24;
          if ( v6 != -4096 && !v7 && v6 != -8192 )
            sub_BD6050((unsigned __int64 *)v8, v23[0] & 0xFFFFFFFFFFFFFFF8LL);
          v8 = v21;
        }
        v21 = v8 + 24;
      }
      if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
        sub_BD60C0(v23);
    }
  }
  v10 = v20;
  v11 = v21;
  v12 = 0;
  if ( v21 != v20 )
  {
    while ( 1 )
    {
      v13 = 0;
      do
      {
        v14 = *((_QWORD *)v10 + 2);
        if ( v14 )
        {
          v15 = sub_29D7CD0(v14, a2);
          if ( (_BYTE)v15 )
            v13 = v15;
        }
        v10 += 24;
      }
      while ( v11 != v10 );
      if ( !(_BYTE)v13 )
        break;
      v10 = v20;
      v11 = v21;
      v12 = v13;
      if ( v21 == v20 )
        goto LABEL_32;
    }
    v17 = v21;
    v10 = v20;
    if ( v21 != v20 )
    {
      do
      {
        v18 = *((_QWORD *)v10 + 2);
        if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
          sub_BD60C0(v10);
        v10 += 24;
      }
      while ( v17 != v10 );
      v10 = v20;
    }
  }
LABEL_32:
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
  return v12;
}
