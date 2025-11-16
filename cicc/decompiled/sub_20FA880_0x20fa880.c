// Function: sub_20FA880
// Address: 0x20fa880
//
void __fastcall sub_20FA880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int v6; // ecx
  _QWORD *v7; // rdi
  int v8; // r12d
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rdx
  int v12; // eax
  _QWORD *v13; // [rsp+0h] [rbp-50h] BYREF
  __int64 v14; // [rsp+8h] [rbp-48h]
  _QWORD v15[8]; // [rsp+10h] [rbp-40h] BYREF

  v6 = 1;
  v7 = v15;
  v8 = 1;
  v13 = v15;
  v15[0] = a2;
  v14 = 0x400000001LL;
  while ( 1 )
  {
    v9 = v7[v6 - 1];
    v10 = *(_QWORD *)(v9 + 32);
    v11 = v10 + 8LL * *(unsigned int *)(v9 + 40);
    if ( v10 != v11 )
      break;
LABEL_10:
    v12 = v14;
    *(_DWORD *)(v9 + 180) = v8;
    v6 = v12 - 1;
    LODWORD(v14) = v12 - 1;
    if ( v12 == 1 )
      goto LABEL_11;
LABEL_9:
    ++v8;
  }
  while ( *(_DWORD *)(*(_QWORD *)v10 + 180LL) )
  {
    v10 += 8;
    if ( v11 == v10 )
      goto LABEL_10;
  }
  if ( v6 >= HIDWORD(v14) )
  {
    sub_16CD150((__int64)&v13, v15, 0, 8, a5, a6);
    v7 = v13;
  }
  v7[(unsigned int)v14] = *(_QWORD *)v10;
  v7 = v13;
  LODWORD(v14) = v14 + 1;
  v6 = v14;
  *(_DWORD *)(*(_QWORD *)v10 + 176LL) = v8;
  if ( v6 )
    goto LABEL_9;
LABEL_11:
  if ( v7 != v15 )
    _libc_free((unsigned __int64)v7);
}
