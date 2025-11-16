// Function: sub_C82D80
// Address: 0xc82d80
//
__int64 __fastcall sub_C82D80(DIR **a1, __int64 a2)
{
  int *v3; // r12
  DIR *v4; // rdi
  struct dirent *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned __int8 *v9; // rbx
  char *d_name; // r14
  size_t v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r9
  __int16 v14; // dx
  __int64 v15; // rdi
  __int64 v16; // r8
  int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  unsigned int v22; // ebx
  _QWORD v23[4]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v24; // [rsp+20h] [rbp-60h]
  __int128 v25; // [rsp+30h] [rbp-50h]
  __int128 v26; // [rsp+40h] [rbp-40h]
  __int128 v27; // [rsp+50h] [rbp-30h]

  v3 = __errno_location();
  do
  {
    while ( 1 )
    {
      *v3 = 0;
      v4 = *a1;
      v5 = readdir(*a1);
      v9 = (unsigned __int8 *)v5;
      if ( !v5 )
      {
        v22 = *v3;
        if ( !*v3 )
          return sub_C82C70((__int64)a1, a2, v6, v7, v8);
        sub_2241E50(v4, a2, v6, v7, v8);
        return v22;
      }
      d_name = v5->d_name;
      v11 = strlen(v5->d_name);
      if ( v11 != 1 )
        break;
      if ( v9[19] != 46 )
        goto LABEL_8;
    }
  }
  while ( v11 == 2 && v9[19] == 46 && v9[20] == 46 );
LABEL_8:
  v14 = v9[18];
  *(_QWORD *)&v27 = 0;
  v15 = (__int64)(a1 + 1);
  *((_QWORD *)&v27 + 1) = 0xFFFF00000000LL;
  v16 = 3;
  v17 = (unsigned __int16)(v14 << 12);
  v25 = 0;
  v26 = 0;
  if ( v17 != 0x4000 )
  {
    v16 = 2;
    if ( v17 != 0x8000 )
    {
      v16 = 5;
      if ( v17 != 24576 )
      {
        v16 = 6;
        if ( v17 != 0x2000 )
        {
          v16 = 7;
          if ( v17 != 4096 )
          {
            v16 = 8;
            if ( v17 != 49152 )
              v16 = 5 * (unsigned int)(v17 != 40960) + 4;
          }
        }
      }
    }
  }
  v24 = 261;
  v23[1] = v11;
  v23[0] = d_name;
  sub_C81FC0(v15, (__int64)v23, v16, v12, v16, v13, v25, v26, v27);
  sub_2241E40(v15, v23, v18, v19, v20);
  return 0;
}
