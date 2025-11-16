// Function: sub_E89FB0
// Address: 0xe89fb0
//
unsigned __int64 __fastcall sub_E89FB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  _DWORD *v3; // r10
  unsigned __int64 result; // rax
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rcx
  unsigned int v7; // r12d
  int v8; // edi
  size_t v9; // rdx
  size_t v10; // rax
  unsigned __int64 *v11; // rcx
  __int64 v12; // [rsp-88h] [rbp-88h]
  size_t v13[4]; // [rsp-78h] [rbp-78h] BYREF
  __int16 v14; // [rsp-58h] [rbp-58h]
  _QWORD v15[4]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v16; // [rsp-28h] [rbp-28h]
  __int64 v17; // [rsp-8h] [rbp-8h]

  v3 = *(_DWORD **)(a1 + 920);
  result = *(_QWORD *)(a1 + 480);
  if ( *v3 == 1 )
  {
    v17 = v2;
    v5 = *(_QWORD *)(a2 + 168) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5 )
    {
      if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
      {
        v11 = *(unsigned __int64 **)(v5 - 8);
        v5 = *v11;
        v6 = v11 + 3;
      }
      else
      {
        v5 = 0;
        v6 = 0;
      }
      v7 = 640;
    }
    else
    {
      v6 = 0;
      v7 = 128;
    }
    v8 = *(_DWORD *)(a2 + 156);
    v15[1] = v5;
    v9 = *(_QWORD *)(result + 128);
    v10 = *(_QWORD *)(result + 136);
    v12 = *(_QWORD *)(a2 + 16);
    v16 = 261;
    v15[0] = v6;
    v14 = 261;
    v13[0] = v9;
    v13[1] = v10;
    return sub_E71CB0((__int64)v3, v13, 1, v7, 0, (__int64)v15, 1u, v8, v12);
  }
  return result;
}
