// Function: sub_ED2120
// Address: 0xed2120
//
__int64 __fastcall sub_ED2120(__int64 a1, unsigned int *a2, unsigned __int64 a3, int a4)
{
  unsigned int v5; // edx
  bool v7; // zf
  unsigned __int32 v8; // ecx
  size_t v9; // r14
  int v10; // esi
  __int64 v11; // rax
  unsigned int *v13; // rax
  unsigned int *v14; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( a3 < (unsigned __int64)(a2 + 2) )
  {
    v10 = 8;
    goto LABEL_6;
  }
  v5 = *a2;
  v7 = a4 == 1;
  v8 = _byteswap_ulong(*a2);
  if ( !v7 )
    v5 = v8;
  v9 = v5;
  if ( a3 < (unsigned __int64)a2 + v5 )
  {
    v10 = 7;
LABEL_6:
    sub_ED07D0(v17, v10);
    v11 = v17[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v11 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v13 = (unsigned int *)sub_22077B0(v5);
  v14 = v13;
  if ( v13 )
    *(_QWORD *)v13 = 0;
  memcpy(v13, a2, v9);
  sub_ED2080(v14, a4);
  sub_ED1F20(v17, v14);
  v15 = v17[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v17[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v15;
    j___libc_free_0(v14);
  }
  else
  {
    v16 = *(unsigned __int8 *)(a1 + 8);
    *(_QWORD *)a1 = v14;
    *(_BYTE *)(a1 + 8) = v16 & 0xFC | 2;
  }
  return a1;
}
