// Function: sub_C45F70
// Address: 0xc45f70
//
__int64 __fastcall sub_C45F70(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  unsigned int v8; // esi
  unsigned __int64 v9; // rdx
  unsigned int v10; // edi
  unsigned __int64 v11; // rcx
  unsigned int v12; // r9d
  char v13; // r8
  __int64 v14; // rax
  char v15; // cl
  char v16; // al
  const void *v18; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-38h]

  v19 = *(_DWORD *)(a2 + 8);
  if ( v19 > 0x40 )
    sub_C43780((__int64)&v18, (const void **)a2);
  else
    v18 = *(const void **)a2;
  sub_C45EE0((__int64)&v18, (__int64 *)a3);
  v8 = v19;
  v9 = (unsigned __int64)v18;
  *(_DWORD *)(a1 + 8) = v19;
  *(_QWORD *)a1 = v9;
  v10 = *(_DWORD *)(a2 + 8);
  if ( v10 > 0x40 )
    v11 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v10 - 1) >> 6));
  else
    v11 = *(_QWORD *)a2;
  v12 = *(_DWORD *)(a3 + 8);
  v13 = (v11 & (1LL << ((unsigned __int8)v10 - 1))) == 0;
  v14 = *(_QWORD *)a3;
  if ( v12 > 0x40 )
    v14 = *(_QWORD *)(v14 + 8LL * ((v12 - 1) >> 6));
  v15 = (v14 & (1LL << ((unsigned __int8)v12 - 1))) == 0;
  v16 = 0;
  if ( v15 == v13 )
  {
    if ( v8 > 0x40 )
      v9 = *(_QWORD *)(v9 + 8LL * ((v8 - 1) >> 6));
    v16 = v13 ^ ((v9 & (1LL << ((unsigned __int8)v8 - 1))) == 0);
  }
  *a4 = v16;
  return a1;
}
