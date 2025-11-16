// Function: sub_108E600
// Address: 0x108e600
//
__int64 __fastcall sub_108E600(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        _QWORD *a6)
{
  unsigned __int64 v6; // rax
  char v11; // si
  int v12; // ecx
  __int64 v13; // rdi
  unsigned __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  unsigned __int32 v19; // eax
  unsigned __int32 v20; // edx
  unsigned __int8 v21[56]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a3;
  v11 = *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL);
  v12 = *(_DWORD *)(a1 + 176);
  v13 = *(_QWORD *)(a1 + 168);
  if ( *(_BYTE *)(*a6 + 150LL) )
  {
    v6 = a6[2] + a3;
    if ( v11 )
    {
LABEL_3:
      v14 = _byteswap_uint64(v6);
      if ( v12 != 1 )
        v6 = v14;
      *(_QWORD *)v21 = v6;
      sub_CB6200(v13, v21, 8u);
      goto LABEL_6;
    }
    v20 = _byteswap_ulong(v6);
    if ( v12 != 1 )
      LODWORD(v6) = v20;
    *(_DWORD *)v21 = v6;
    sub_CB6200(v13, v21, 4u);
  }
  else
  {
    if ( v11 )
      goto LABEL_3;
    v19 = _byteswap_ulong(a3);
    if ( v12 != 1 )
      a3 = v19;
    *(_DWORD *)v21 = a3;
    sub_CB6200(v13, v21, 4u);
  }
LABEL_6:
  v15 = *(_QWORD *)(a1 + 168);
  if ( *(_DWORD *)(a1 + 176) != 1 )
    a2 = _byteswap_ulong(a2);
  *(_DWORD *)v21 = a2;
  sub_CB6200(v15, v21, 4u);
  v16 = *(_QWORD *)(a1 + 168);
  v21[0] = a4;
  sub_CB6200(v16, v21, 1u);
  v17 = *(_QWORD *)(a1 + 168);
  v21[0] = a5;
  return sub_CB6200(v17, v21, 1u);
}
