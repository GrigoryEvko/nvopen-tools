// Function: sub_C83650
// Address: 0xc83650
//
__int64 __fastcall sub_C83650(__int64 a1, unsigned int a2, void *a3, size_t a4, __off_t a5)
{
  __off_t v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  int *v13; // r14
  int v14; // r15d
  ssize_t v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  char v19; // bl
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = lseek(a2, a5, 0);
  v13 = __errno_location();
  if ( v9 == -1 )
  {
    v21 = sub_2241E50(a2, a5, v10, v11, v12);
    sub_C63CA0(v25, *v13, v21);
    v22 = v25[0] & 0xFFFFFFFFFFFFFFFELL;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v22;
  }
  else
  {
    do
    {
      *v13 = 0;
      v15 = read(a2, a3, a4);
      if ( v15 != -1 )
      {
        v19 = *(_BYTE *)(a1 + 8);
        *(_QWORD *)a1 = v15;
        *(_BYTE *)(a1 + 8) = v19 & 0xFC | 2;
        return a1;
      }
      v14 = *v13;
    }
    while ( *v13 == 4 );
    v23 = sub_2241E50(a2, a3, v16, v17, v18);
    sub_C63CA0(v25, v14, v23);
    v24 = v25[0] & 0xFFFFFFFFFFFFFFFELL;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v24;
  }
  return a1;
}
