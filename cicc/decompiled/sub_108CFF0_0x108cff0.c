// Function: sub_108CFF0
// Address: 0x108cff0
//
__int64 __fastcall sub_108CFF0(__int64 a1, __int64 a2, unsigned __int8 a3, unsigned __int8 a4)
{
  unsigned __int32 v4; // eax
  bool v8; // zf
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  unsigned __int32 v15; // eax
  __int64 v16; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  unsigned __int8 v20[52]; // [rsp+Ch] [rbp-34h] BYREF

  v4 = a2;
  v8 = *(_DWORD *)(a1 + 176) == 1;
  v9 = *(_QWORD *)(a1 + 168);
  if ( !v8 )
    v4 = _byteswap_ulong(a2);
  *(_DWORD *)v20 = v4;
  sub_CB6200(v9, v20, 4u);
  v10 = *(_QWORD *)(a1 + 168);
  *(_DWORD *)v20 = 0;
  sub_CB6200(v10, v20, 4u);
  v11 = *(_QWORD *)(a1 + 168);
  *(_WORD *)v20 = 0;
  sub_CB6200(v11, v20, 2u);
  v12 = *(_QWORD *)(a1 + 168);
  v20[0] = a3;
  sub_CB6200(v12, v20, 1u);
  v13 = *(_QWORD *)(a1 + 168);
  v20[0] = a4;
  sub_CB6200(v13, v20, 1u);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 184) + 8LL) )
  {
    v14 = *(_QWORD *)(a1 + 168);
    v15 = HIDWORD(a2);
    if ( *(_DWORD *)(a1 + 176) != 1 )
      v15 = _byteswap_ulong(HIDWORD(a2));
    *(_DWORD *)v20 = v15;
    sub_CB6200(v14, v20, 4u);
    sub_CB6C70(*(_QWORD *)(a1 + 168), 1u);
    v16 = *(_QWORD *)(a1 + 168);
    v20[0] = -5;
    return sub_CB6200(v16, v20, 1u);
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 168);
    *(_DWORD *)v20 = 0;
    sub_CB6200(v18, v20, 4u);
    v19 = *(_QWORD *)(a1 + 168);
    *(_WORD *)v20 = 0;
    return sub_CB6200(v19, v20, 2u);
  }
}
