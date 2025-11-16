// Function: sub_1673EB0
// Address: 0x1673eb0
//
bool __fastcall sub_1673EB0(__int64 a1, __int64 **a2, __int64 a3, const void *a4, __int64 a5)
{
  unsigned __int64 v7; // rdx
  void *v8; // r8
  size_t v9; // r14
  int v10; // eax
  char *v12; // rdi
  void *src; // [rsp+8h] [rbp-58h]
  char *v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h]
  _BYTE v16[64]; // [rsp+20h] [rbp-40h] BYREF

  sub_1643FB0((__int64)a2, a4, a5, (*(_DWORD *)(a3 + 8) & 0x200) != 0);
  if ( !*(_QWORD *)(a3 + 24) )
    return sub_1673BC0(*(_QWORD *)(a1 + 640), (__int64)a2);
  v14 = v16;
  v8 = (void *)sub_1643640(a3);
  v15 = 0x1000000000LL;
  v9 = v7;
  v10 = v7;
  if ( v7 > 0x10 )
  {
    src = v8;
    sub_16CD150(&v14, v16, v7, 1);
    v8 = src;
    v12 = &v14[(unsigned int)v15];
  }
  else
  {
    if ( !v7 )
      goto LABEL_4;
    v12 = v16;
  }
  memcpy(v12, v8, v9);
  v10 = v9 + v15;
LABEL_4:
  LODWORD(v15) = v10;
  sub_1643660((__int64 **)a3, byte_3F871B3, 0);
  sub_1643660(a2, v14, (unsigned int)v15);
  if ( v14 != v16 )
    _libc_free((unsigned __int64)v14);
  return sub_1673BC0(*(_QWORD *)(a1 + 640), (__int64)a2);
}
