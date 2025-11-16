// Function: sub_11FD5F0
// Address: 0x11fd5f0
//
__int64 __fastcall sub_11FD5F0(void **a1)
{
  char *v1; // rax
  unsigned __int8 *v2; // rcx
  int v4; // esi
  unsigned __int8 *v5; // rax
  int v6; // edx
  unsigned __int64 v7; // r12
  unsigned __int8 *v9; // rsi
  int v10; // edi
  unsigned __int8 *v11; // rax
  int v12; // edx
  unsigned __int8 *v13; // rcx
  unsigned __int64 v14; // r13
  void *v15; // rax
  void *v16; // r12
  int v17; // edx
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  void **v20; // rbx
  void *v21; // [rsp-48h] [rbp-48h] BYREF
  void **v22; // [rsp-40h] [rbp-40h]

  v1 = (char *)*a1;
  if ( (unsigned int)*(unsigned __int8 *)*a1 - 48 > 9 )
    return 1;
  v2 = (unsigned __int8 *)(v1 + 1);
  *a1 = v1 + 1;
  v4 = (unsigned __int8)v1[1];
  if ( (unsigned int)(v4 - 48) <= 9 )
  {
    v5 = (unsigned __int8 *)(v1 + 2);
    do
    {
      *a1 = v5;
      v6 = *v5;
      v2 = v5++;
      LOBYTE(v4) = v6;
    }
    while ( (unsigned int)(v6 - 48) <= 9 );
  }
  v7 = (unsigned __int64)a1[7];
  if ( (_BYTE)v4 == 46 )
  {
    v9 = v2 + 1;
    *a1 = v2 + 1;
    v10 = v2[1];
    if ( (unsigned int)(v10 - 48) <= 9 )
    {
      v11 = v2 + 2;
      do
      {
        *a1 = v11;
        v12 = *v11;
        v9 = v11++;
        LOBYTE(v10) = v12;
      }
      while ( (unsigned int)(v12 - 48) <= 9 );
    }
    if ( (v10 & 0xDF) != 0x45
      || (v17 = v9[1], (unsigned int)(v17 - 48) > 9)
      && ((((_BYTE)v17 - 43) & 0xFD) != 0 || (unsigned int)v9[2] - 48 > 9) )
    {
      v13 = (unsigned __int8 *)*a1;
    }
    else
    {
      v13 = v9 + 2;
      *a1 = v9 + 2;
      if ( (unsigned int)v9[2] - 48 <= 9 )
      {
        v18 = v9 + 3;
        do
        {
          v13 = v18;
          *a1 = v18++;
        }
        while ( (unsigned int)*v13 - 48 <= 9 );
      }
    }
    v14 = (unsigned __int64)&v13[-v7];
    v15 = sub_C33320();
    sub_C43310(&v21, v15, v7, v14);
    sub_11FD130(a1 + 15, &v21);
    v16 = sub_C33340();
    if ( v21 == v16 )
    {
      if ( v22 )
      {
        v19 = 3LL * (_QWORD)*(v22 - 1);
        v20 = &v22[v19];
        while ( v22 != v20 )
        {
          v20 -= 3;
          if ( v16 == *v20 )
            sub_969EE0((__int64)v20);
          else
            sub_C338F0((__int64)v20);
        }
        j_j_j___libc_free_0_0(v20 - 1);
      }
    }
    else
    {
      sub_C338F0((__int64)&v21);
    }
    return 528;
  }
  else
  {
    *a1 = (void *)(v7 + 1);
    return 1;
  }
}
