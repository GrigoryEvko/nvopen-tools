// Function: sub_3351190
// Address: 0x3351190
//
void __fastcall sub_3351190(__int64 a1)
{
  __int64 v2; // rdi
  char *v3; // rsi
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  char *v7; // rax
  char *v8; // rsi
  __int64 v9; // rcx
  unsigned __int64 v10; // rax
  char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16[2]; // [rsp+8h] [rbp-18h] BYREF

  v16[0] = 0;
  *(_DWORD *)(a1 + 776) = 0;
  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(char **)(a1 + 792);
  v4 = *(_QWORD *)(a1 + 784);
  v5 = *(unsigned int *)(v2 + 16);
  v6 = (__int64)&v3[-v4] >> 3;
  if ( v5 > v6 )
  {
    sub_334D6D0(a1 + 784, v3, v5 - v6, v16);
    v5 = *(unsigned int *)(*(_QWORD *)(a1 + 24) + 16LL);
  }
  else if ( v5 < v6 )
  {
    v7 = (char *)(v4 + 8 * v5);
    if ( v3 != v7 )
    {
      *(_QWORD *)(a1 + 792) = v7;
      v5 = *(unsigned int *)(v2 + 16);
    }
  }
  v8 = *(char **)(a1 + 816);
  v9 = *(_QWORD *)(a1 + 808);
  LODWORD(v16[0]) = 0;
  v10 = (__int64)&v8[-v9] >> 2;
  if ( v10 < v5 )
  {
    sub_1CFD340(a1 + 808, v8, v5 - v10, v16);
  }
  else if ( v10 > v5 )
  {
    v11 = (char *)(v9 + 4 * v5);
    if ( v8 != v11 )
      *(_QWORD *)(a1 + 816) = v11;
  }
  sub_3360940(a1);
  sub_334EC10(a1, (__int64)v8, v12, v13, v14, v15);
}
