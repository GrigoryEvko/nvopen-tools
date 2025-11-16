// Function: sub_108CE60
// Address: 0x108ce60
//
__int64 __fastcall sub_108CE60(__int64 a1, char **a2, char a3)
{
  __int64 v4; // rdi
  char *v5; // r13
  __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rdi
  unsigned __int32 v10; // edx
  __int64 v11; // rdi
  __int64 v12; // rdi
  char *v14; // r13
  __int64 v15; // r14
  unsigned __int8 v16; // al
  unsigned __int8 v18; // [rsp+1Fh] [rbp-41h] BYREF
  char dest[14]; // [rsp+20h] [rbp-40h] BYREF
  char v20; // [rsp+2Eh] [rbp-32h] BYREF

  if ( (unsigned __int64)a2[1] <= 0xE )
  {
    v14 = dest;
    strncpy(dest, *a2, 0xEu);
    v15 = *(_QWORD *)(a1 + 168);
    do
    {
      v16 = *v14++;
      v18 = v16;
      sub_CB6200(v15, &v18, 1u);
    }
    while ( v14 != &v20 );
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 168);
    *(_DWORD *)dest = 0;
    sub_CB6200(v4, (unsigned __int8 *)dest, 4u);
    v5 = *a2;
    v6 = (__int64)a2[1];
    v7 = sub_C94890(*a2, v6);
    v8 = sub_C0C3A0(a1 + 192, v5, (v7 << 32) | (unsigned int)v6);
    v9 = *(_QWORD *)(a1 + 168);
    v10 = v8;
    if ( *(_DWORD *)(a1 + 176) != 1 )
      v10 = _byteswap_ulong(v8);
    *(_DWORD *)dest = v10;
    sub_CB6200(v9, (unsigned __int8 *)dest, 4u);
    sub_CB6C70(*(_QWORD *)(a1 + 168), 6u);
  }
  v11 = *(_QWORD *)(a1 + 168);
  dest[0] = a3;
  sub_CB6200(v11, (unsigned __int8 *)dest, 1u);
  sub_CB6C70(*(_QWORD *)(a1 + 168), 2u);
  v12 = *(_QWORD *)(a1 + 168);
  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
    return sub_CB6C70(v12, 1u);
  dest[0] = -4;
  return sub_CB6200(v12, (unsigned __int8 *)dest, 1u);
}
