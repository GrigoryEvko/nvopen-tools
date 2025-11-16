// Function: sub_3886990
// Address: 0x3886990
//
__int64 __fastcall sub_3886990(unsigned __int8 **a1)
{
  char *v2; // r14
  unsigned __int8 *v3; // r12
  int v4; // ebx
  unsigned __int8 *v5; // rax
  int v6; // edi
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // r15
  unsigned __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // rdx
  unsigned __int8 *v12; // rbx
  unsigned __int8 *v13; // rdx
  int v14; // ecx
  unsigned __int8 *v15; // r12
  int v16; // eax
  int v17; // ecx
  unsigned __int8 *v18; // rcx
  unsigned __int8 *v19; // rdx
  __int64 v20; // r12
  void *v21; // rax
  void *v22; // rbx
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // [rsp+0h] [rbp-60h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v34; // [rsp+10h] [rbp-50h] BYREF
  void *v35; // [rsp+18h] [rbp-48h] BYREF
  __int64 v36; // [rsp+20h] [rbp-40h]

  v2 = (char *)a1[6];
  v3 = *a1;
  v4 = **a1;
  if ( (unsigned int)(unsigned __int8)*v2 - 48 <= 9 )
  {
    v6 = (unsigned __int8)v4;
    if ( (unsigned int)(unsigned __int8)v4 - 48 > 9 )
      goto LABEL_5;
  }
  else if ( (unsigned int)(v4 - 48) > 9 )
  {
    v12 = sub_3880AC0(*a1);
    result = 1;
    if ( v12 )
    {
      sub_2241130((unsigned __int64 *)a1 + 8, 0, (unsigned __int64)a1[9], v2, v12 - 1 - (unsigned __int8 *)v2);
      *a1 = v12;
      return 372;
    }
    return result;
  }
  v5 = v3 + 1;
  do
  {
    *a1 = v5;
    v6 = *v5;
    v3 = v5++;
    v4 = v6;
  }
  while ( (unsigned int)(v6 - 48) <= 9 );
LABEL_5:
  if ( !isalnum(v6) )
  {
    if ( (unsigned __int8)(v4 - 36) > 0x3Bu || (v11 = 0x800000000400601LL, !_bittest64(&v11, (unsigned int)(v4 - 36))) )
    {
LABEL_8:
      if ( *v2 == 48 && v2[1] == 120 )
        return sub_3886120((__int64)a1);
      sub_39457C0(&v34, v2, v3 - (unsigned __int8 *)v2);
      if ( *((_DWORD *)a1 + 38) > 0x40u )
      {
        v9 = (unsigned __int64)a1[18];
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
      a1[18] = v34;
      *((_DWORD *)a1 + 38) = (_DWORD)v35;
      *((_BYTE *)a1 + 156) = BYTE4(v35);
      return 390;
    }
    v7 = sub_3880AC0(v3);
    v8 = v7;
    if ( !v7 )
      goto LABEL_7;
LABEL_32:
    sub_2241130((unsigned __int64 *)a1 + 8, 0, (unsigned __int64)a1[9], v2, v7 - 1 - (unsigned __int8 *)v2);
    *a1 = v8;
    return 372;
  }
  v7 = sub_3880AC0(v3);
  v8 = v7;
  if ( v7 )
    goto LABEL_32;
LABEL_7:
  if ( (_BYTE)v4 != 46 )
    goto LABEL_8;
  v13 = v3 + 1;
  *a1 = v3 + 1;
  v14 = v3[1];
  if ( (unsigned int)(v14 - 48) <= 9 )
  {
    v15 = v3 + 2;
    do
    {
      *a1 = v15;
      v16 = *v15;
      v13 = v15++;
      LOBYTE(v14) = v16;
    }
    while ( (unsigned int)(v16 - 48) <= 9 );
  }
  if ( (v14 & 0xDF) == 0x45
    && ((v17 = v13[1], (unsigned int)(v17 - 48) <= 9)
     || (((_BYTE)v17 - 43) & 0xFD) == 0 && (unsigned int)v13[2] - 48 <= 9) )
  {
    v18 = v13 + 2;
    *a1 = v13 + 2;
    if ( (unsigned int)v13[2] - 48 <= 9 )
    {
      v19 = v13 + 3;
      do
      {
        v18 = v19;
        *a1 = v19++;
      }
      while ( (unsigned int)*v18 - 48 <= 9 );
    }
  }
  else
  {
    v18 = *a1;
  }
  v20 = v18 - (unsigned __int8 *)v2;
  v21 = sub_1698280();
  sub_169E660((__int64)&v34, v21, v2, v20);
  sub_3880CD0((void **)a1 + 15, &v35);
  v22 = sub_16982C0();
  if ( v35 == v22 )
  {
    v23 = v36;
    if ( v36 )
    {
      v24 = 32LL * *(_QWORD *)(v36 - 8);
      v25 = v36 + v24;
      if ( v36 != v36 + v24 )
      {
        do
        {
          v25 -= 32;
          if ( v22 == *(void **)(v25 + 8) )
          {
            v26 = *(_QWORD *)(v25 + 16);
            if ( v26 )
            {
              v27 = 32LL * *(_QWORD *)(v26 - 8);
              v28 = v26 + v27;
              while ( v26 != v28 )
              {
                v28 -= 32;
                if ( v22 == *(void **)(v28 + 8) )
                {
                  v29 = *(_QWORD *)(v28 + 16);
                  if ( v29 )
                  {
                    v30 = 32LL * *(_QWORD *)(v29 - 8);
                    v31 = v29 + v30;
                    if ( v29 != v29 + v30 )
                    {
                      do
                      {
                        v32 = v29;
                        v33 = v31 - 32;
                        sub_127D120((_QWORD *)(v31 - 24));
                        v31 = v33;
                        v29 = v32;
                      }
                      while ( v32 != v33 );
                    }
                    j_j_j___libc_free_0_0(v29 - 8);
                  }
                }
                else
                {
                  sub_1698460(v28 + 8);
                }
              }
              j_j_j___libc_free_0_0(v26 - 8);
            }
          }
          else
          {
            sub_1698460(v25 + 8);
          }
        }
        while ( v23 != v25 );
      }
      j_j_j___libc_free_0_0(v23 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v35);
  }
  return 389;
}
