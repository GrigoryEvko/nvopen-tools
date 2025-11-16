// Function: sub_E89E30
// Address: 0xe89e30
//
unsigned __int64 __fastcall sub_E89E30(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  _DWORD *v3; // r10
  unsigned __int64 v4; // rax
  unsigned __int64 *v5; // rdx
  _QWORD *v6; // rdx
  unsigned int v7; // ecx
  int v9; // edi
  __int64 v10; // [rsp-78h] [rbp-78h]
  const char *v11; // [rsp-68h] [rbp-68h] BYREF
  char v12; // [rsp-48h] [rbp-48h]
  char v13; // [rsp-47h] [rbp-47h]
  _QWORD v14[4]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v15; // [rsp-18h] [rbp-18h]
  __int64 v16; // [rsp-8h] [rbp-8h]

  v3 = *(_DWORD **)(a1 + 920);
  if ( *v3 != 1 || v3[14] == 39 && v3[16] == 3 && v3[17] == 24 )
    return *(_QWORD *)(a1 + 472);
  v16 = v2;
  v4 = *(_QWORD *)(a2 + 168) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 )
  {
    if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
    {
      v5 = *(unsigned __int64 **)(v4 - 8);
      v4 = *v5;
      v6 = v5 + 3;
    }
    else
    {
      v4 = 0;
      v6 = 0;
    }
    v7 = 640;
  }
  else
  {
    v6 = 0;
    v7 = 128;
  }
  v14[0] = v6;
  v9 = *(_DWORD *)(a2 + 156);
  v10 = *(_QWORD *)(a2 + 16);
  v15 = 261;
  v14[1] = v4;
  v13 = 1;
  v11 = ".stack_sizes";
  v12 = 3;
  return sub_E71CB0((__int64)v3, (size_t *)&v11, 1, v7, 0, (__int64)v14, 1u, v9, v10);
}
