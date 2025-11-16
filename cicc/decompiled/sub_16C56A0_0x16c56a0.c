// Function: sub_16C56A0
// Address: 0x16c56a0
//
__int64 __fastcall sub_16C56A0(_QWORD *a1)
{
  char *v2; // rax
  const char *v3; // r12
  size_t v4; // rsi
  const char *v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // r13d
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  const char *v15; // rsi
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 v19; // rdx
  char *v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r8
  size_t v23; // r13
  __int64 v24; // rax
  size_t v25; // rdx
  char *v26; // [rsp+0h] [rbp-110h] BYREF
  __int16 v27; // [rsp+10h] [rbp-100h]
  const char *v28; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v29; // [rsp+30h] [rbp-E0h]
  char *v30; // [rsp+40h] [rbp-D0h] BYREF
  char v31; // [rsp+50h] [rbp-C0h]
  char v32; // [rsp+51h] [rbp-BFh]
  _OWORD v33[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int128 v34; // [rsp+80h] [rbp-90h]
  __int128 v35; // [rsp+90h] [rbp-80h]
  _OWORD v36[2]; // [rsp+A0h] [rbp-70h] BYREF
  __int128 v37; // [rsp+C0h] [rbp-50h]
  __int128 v38; // [rsp+D0h] [rbp-40h]

  *((_DWORD *)a1 + 2) = 0;
  v2 = getenv("PWD");
  v34 = 0;
  v37 = 0;
  DWORD1(v34) = 0xFFFF;
  DWORD1(v37) = 0xFFFF;
  memset(v33, 0, sizeof(v33));
  v35 = 0;
  memset(v36, 0, sizeof(v36));
  v38 = 0;
  if ( !v2 )
    goto LABEL_5;
  v3 = v2;
  v27 = 257;
  if ( *v2 )
  {
    v26 = v2;
    LOBYTE(v27) = 3;
  }
  if ( !(unsigned __int8)sub_16C4E60((__int64)&v26, 2u) )
    goto LABEL_5;
  v29 = 257;
  if ( *v3 )
  {
    v28 = v3;
    LOBYTE(v29) = 3;
  }
  if ( !(unsigned int)sub_16C55F0((__int64)&v28, (__int64)v33, 1)
    && (v32 = 1, v15 = (const char *)v36, v30 = ".", v31 = 3, !(unsigned int)sub_16C55F0((__int64)&v30, (__int64)v36, 1))
    && (v16 = sub_16C4FE0((__int64)v36), v18 = v17, sub_16C4FE0((__int64)v33) == v16)
    && v19 == v18 )
  {
    v20 = (char *)v3;
    v23 = strlen(v3);
    v24 = *((unsigned int *)a1 + 2);
    v25 = *((unsigned int *)a1 + 3) - v24;
    if ( v23 > v25 )
    {
      v15 = (const char *)(a1 + 2);
      v20 = (char *)a1;
      sub_16CD150(a1, a1 + 2, v23 + v24, 1);
      v24 = *((unsigned int *)a1 + 2);
    }
    if ( v23 )
    {
      v15 = v3;
      v20 = (char *)(*a1 + v24);
      memcpy(v20, v3, v23);
      LODWORD(v24) = *((_DWORD *)a1 + 2);
    }
    *((_DWORD *)a1 + 2) = v23 + v24;
    sub_2241E40(v20, v15, v25, v21, v22);
    return 0;
  }
  else
  {
LABEL_5:
    v4 = *((unsigned int *)a1 + 3);
    if ( (unsigned int)v4 <= 0xFFF )
    {
      sub_16CD150(a1, a1 + 2, 4096, 1);
      v4 = *((unsigned int *)a1 + 3);
    }
LABEL_7:
    v5 = (const char *)*a1;
    if ( getcwd((char *)*a1, v4) )
    {
LABEL_11:
      v10 = *a1;
      *((_DWORD *)a1 + 2) = strlen((const char *)*a1);
      sub_2241E40(v10, v4, v11, v12, v13);
      return 0;
    }
    else
    {
      while ( 1 )
      {
        v9 = *__errno_location();
        if ( v9 != 12 )
          break;
        v4 = *((unsigned int *)a1 + 3);
        if ( 2 * v4 <= v4 )
          goto LABEL_7;
        sub_16CD150(a1, a1 + 2, 2 * v4, 1);
        v4 = *((unsigned int *)a1 + 3);
        v5 = (const char *)*a1;
        if ( getcwd((char *)*a1, v4) )
          goto LABEL_11;
      }
      sub_2241E50(v5, v4, v6, v7, v8);
      return v9;
    }
  }
}
