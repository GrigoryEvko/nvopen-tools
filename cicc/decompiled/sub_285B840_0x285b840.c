// Function: sub_285B840
// Address: 0x285b840
//
void __fastcall sub_285B840(__int64 a1, _QWORD *a2, __int64 a3, __int64 *a4)
{
  unsigned __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rdx
  _BYTE **v9; // r9
  bool v10; // al
  int v11; // eax
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // [rsp+8h] [rbp-98h]
  _BYTE *v16; // [rsp+10h] [rbp-90h] BYREF
  __int64 v17; // [rsp+18h] [rbp-88h]
  _BYTE v18[32]; // [rsp+20h] [rbp-80h] BYREF
  _BYTE *v19; // [rsp+40h] [rbp-60h] BYREF
  __int64 v20; // [rsp+48h] [rbp-58h]
  _BYTE v21[80]; // [rsp+50h] [rbp-50h] BYREF

  v16 = v18;
  v17 = 0x400000000LL;
  v19 = v21;
  v20 = 0x400000000LL;
  sub_285B280(a2, a3, (__int64)&v16, (__int64)&v19, a4);
  v8 = (unsigned int)v17;
  v9 = &v16;
  if ( !(_DWORD)v17 )
  {
    if ( !(_DWORD)v20 )
      goto LABEL_3;
    goto LABEL_10;
  }
  v15 = sub_DC7EB0(a4, (__int64)&v16, 0, 0);
  v10 = sub_D968A0((__int64)v15);
  v7 = (__int64)v15;
  if ( !v10 )
  {
    v13 = *(unsigned int *)(a1 + 48);
    v6 = *(unsigned int *)(a1 + 52);
    if ( v13 + 1 > v6 )
    {
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v13 + 1, 8u, (__int64)v15, (__int64)v9);
      v13 = *(unsigned int *)(a1 + 48);
      v7 = (__int64)v15;
    }
    v8 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(v8 + 8 * v13) = v7;
    ++*(_DWORD *)(a1 + 48);
  }
  v11 = v20;
  *(_BYTE *)(a1 + 24) = 1;
  if ( v11 )
  {
LABEL_10:
    v12 = sub_DC7EB0(a4, (__int64)&v19, 0, 0);
    if ( !sub_D968A0((__int64)v12) )
    {
      v14 = *(unsigned int *)(a1 + 48);
      v6 = *(unsigned int *)(a1 + 52);
      if ( v14 + 1 > v6 )
      {
        sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v14 + 1, 8u, v7, (__int64)v9);
        v14 = *(unsigned int *)(a1 + 48);
      }
      v8 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(v8 + 8 * v14) = v12;
      ++*(_DWORD *)(a1 + 48);
    }
    *(_BYTE *)(a1 + 24) = 1;
  }
LABEL_3:
  sub_2857080(a1, a3, v8, v6, v7, (__int64)v9);
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
  if ( v16 != v18 )
    _libc_free((unsigned __int64)v16);
}
