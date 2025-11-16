// Function: sub_C835A0
// Address: 0xc835a0
//
__int64 __fastcall sub_C835A0(__int64 a1, unsigned int a2, void *a3, size_t a4)
{
  int *v7; // r15
  __int64 v8; // rsi
  ssize_t v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  char v13; // dl
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // [rsp+Ch] [rbp-44h]
  __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = __errno_location();
  do
  {
    *v7 = 0;
    v9 = read(a2, a3, a4);
    if ( v9 != -1 )
    {
      v13 = *(_BYTE *)(a1 + 8);
      *(_QWORD *)a1 = v9;
      *(_BYTE *)(a1 + 8) = v13 & 0xFC | 2;
      return a1;
    }
    v8 = (unsigned int)*v7;
  }
  while ( (_DWORD)v8 == 4 );
  v17 = *v7;
  v15 = sub_2241E50(a2, v8, v10, v11, v12);
  sub_C63CA0(v18, v17, v15);
  v16 = v18[0];
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v16 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
