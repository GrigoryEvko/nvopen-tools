// Function: sub_11FEC70
// Address: 0x11fec70
//
__int64 __fastcall sub_11FEC70(__int64 a1)
{
  char *v2; // r14
  unsigned __int8 *v3; // r12
  int v4; // r15d
  unsigned __int8 *v5; // rax
  int v6; // edi
  int v7; // ebx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned __int8 *v10; // r9
  __int64 v11; // rdi
  __int64 result; // rax
  unsigned __int8 *v13; // rbx
  __int64 v14; // rdx
  unsigned __int8 *v15; // rdx
  int v16; // ecx
  unsigned __int8 *v17; // r12
  int v18; // eax
  int v19; // ecx
  unsigned __int8 *v20; // rcx
  unsigned __int8 *v21; // rdx
  unsigned __int64 v22; // r12
  void *v23; // rax
  const char *v24; // r12
  unsigned __int64 v25; // rax
  int v26; // ebx
  unsigned __int64 v27; // rsi
  __int64 v28; // rax
  void **v29; // rbx
  unsigned __int8 *v30; // [rsp+8h] [rbp-68h]
  const char *v31; // [rsp+10h] [rbp-60h] BYREF
  void **v32; // [rsp+18h] [rbp-58h]
  char v33; // [rsp+30h] [rbp-40h]
  char v34; // [rsp+31h] [rbp-3Fh]

  v2 = *(char **)(a1 + 56);
  v3 = *(unsigned __int8 **)a1;
  v4 = (unsigned __int8)*v2;
  if ( (unsigned int)(v4 - 48) <= 9 )
  {
    v6 = *v3;
    v7 = v6;
    if ( (unsigned int)(v6 - 48) > 9 )
      goto LABEL_6;
  }
  else if ( (unsigned int)*v3 - 48 > 9 )
  {
    v13 = sub_11FD0C0(*(unsigned __int8 **)a1);
    result = 1;
    if ( v13 )
    {
      sub_2241130(a1 + 72, 0, *(_QWORD *)(a1 + 80), v2, v13 - 1 - (unsigned __int8 *)v2);
      *(_QWORD *)a1 = v13;
      return 507;
    }
    return result;
  }
  v5 = v3 + 1;
  do
  {
    *(_QWORD *)a1 = v5;
    v6 = *v5;
    v3 = v5++;
    v7 = v6;
  }
  while ( (unsigned int)(v6 - 48) <= 9 );
  LOBYTE(v4) = *v2;
LABEL_6:
  if ( (unsigned int)((char)v4 - 48) <= 9 && (_BYTE)v7 == 58 )
  {
    v25 = sub_11FE300(a1, v2, (char *)v3);
    ++*(_QWORD *)a1;
    v26 = v25;
    if ( v25 != (unsigned int)v25 )
    {
      v27 = *(_QWORD *)(a1 + 56);
      v34 = 1;
      v31 = "invalid value number (too large)";
      v33 = 3;
      sub_11FD800(a1, v27, (__int64)&v31, 2);
    }
    *(_DWORD *)(a1 + 104) = v26;
    return 502;
  }
  else
  {
    if ( !isalnum(v6) )
    {
      if ( (unsigned __int8)(v7 - 36) > 0x3Bu || (v14 = 0x800000000000601LL, !_bittest64(&v14, (unsigned int)(v7 - 36))) )
      {
        if ( (_BYTE)v7 == 58 )
        {
          v10 = sub_11FD0C0(v3);
          if ( v10 )
            goto LABEL_23;
        }
LABEL_11:
        if ( (_BYTE)v4 == 48 && v2[1] == 120 )
          return sub_11FE690(a1);
        sub_1254190(&v31, v2, v3 - (unsigned __int8 *)v2, v8, v9, v10);
        if ( *(_DWORD *)(a1 + 152) > 0x40u )
        {
          v11 = *(_QWORD *)(a1 + 144);
          if ( v11 )
            j_j___libc_free_0_0(v11);
        }
        *(_QWORD *)(a1 + 144) = v31;
        *(_DWORD *)(a1 + 152) = (_DWORD)v32;
        *(_BYTE *)(a1 + 156) = BYTE4(v32);
        return 529;
      }
    }
    v10 = sub_11FD0C0(v3);
    if ( v10 )
    {
LABEL_23:
      v30 = v10;
      sub_2241130(a1 + 72, 0, *(_QWORD *)(a1 + 80), v2, v10 - 1 - (unsigned __int8 *)v2);
      *(_QWORD *)a1 = v30;
      return 507;
    }
    if ( (_BYTE)v7 != 46 )
      goto LABEL_11;
    v15 = v3 + 1;
    *(_QWORD *)a1 = v3 + 1;
    v16 = v3[1];
    if ( (unsigned int)(v16 - 48) <= 9 )
    {
      v17 = v3 + 2;
      do
      {
        *(_QWORD *)a1 = v17;
        v18 = *v17;
        v15 = v17++;
        LOBYTE(v16) = v18;
      }
      while ( (unsigned int)(v18 - 48) <= 9 );
    }
    if ( (v16 & 0xDF) == 0x45
      && ((v19 = v15[1], (unsigned int)(v19 - 48) <= 9)
       || (((_BYTE)v19 - 43) & 0xFD) == 0 && (unsigned int)v15[2] - 48 <= 9) )
    {
      v20 = v15 + 2;
      *(_QWORD *)a1 = v15 + 2;
      if ( (unsigned int)v15[2] - 48 <= 9 )
      {
        v21 = v15 + 3;
        do
        {
          v20 = v21;
          *(_QWORD *)a1 = v21++;
        }
        while ( (unsigned int)*v20 - 48 <= 9 );
      }
    }
    else
    {
      v20 = *(unsigned __int8 **)a1;
    }
    v22 = v20 - (unsigned __int8 *)v2;
    v23 = sub_C33320();
    sub_C43310((void **)&v31, v23, (unsigned __int64)v2, v22);
    sub_11FD130((void **)(a1 + 120), (void **)&v31);
    v24 = (const char *)sub_C33340();
    if ( v31 == v24 )
    {
      if ( v32 )
      {
        v28 = 3LL * (_QWORD)*(v32 - 1);
        v29 = &v32[v28];
        while ( v32 != v29 )
        {
          v29 -= 3;
          if ( v24 == *v29 )
            sub_969EE0((__int64)v29);
          else
            sub_C338F0((__int64)v29);
        }
        j_j_j___libc_free_0_0(v29 - 1);
      }
    }
    else
    {
      sub_C338F0((__int64)&v31);
    }
    return 528;
  }
}
