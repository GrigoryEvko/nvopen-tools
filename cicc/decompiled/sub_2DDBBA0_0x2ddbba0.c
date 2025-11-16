// Function: sub_2DDBBA0
// Address: 0x2ddbba0
//
void __fastcall sub_2DDBBA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rdx
  _DWORD *v8; // rax
  unsigned int *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdi
  unsigned __int64 v12[2]; // [rsp+0h] [rbp-50h] BYREF
  _BYTE v13[64]; // [rsp+10h] [rbp-40h] BYREF

  v6 = a1;
  v7 = *(unsigned int *)(a1 + 8);
  v12[1] = 0x400000000LL;
  v8 = v13;
  v12[0] = (unsigned __int64)v13;
  if ( (_DWORD)v7 )
  {
    sub_2DDB710((__int64)v12, a1, v7, a4, a5, a6);
    v8 = (_DWORD *)v12[0];
  }
  while ( 1 )
  {
    v9 = *(unsigned int **)(v6 - 48);
    v10 = *v9;
    if ( *v8 >= (unsigned int)v10 && (*v8 != (_DWORD)v10 || v8[1] >= v9[1]) )
      break;
    v11 = v6;
    v6 -= 48;
    sub_2DDB710(v11, v6, (__int64)v9, v10, a5, a6);
    v8 = (_DWORD *)v12[0];
  }
  sub_2DDB710(v6, (__int64)v12, (__int64)v9, v10, a5, a6);
  if ( (_BYTE *)v12[0] != v13 )
    _libc_free(v12[0]);
}
