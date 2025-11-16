// Function: sub_124CF30
// Address: 0x124cf30
//
__int64 __fastcall sub_124CF30(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int8 a6,
        unsigned int a7,
        char a8)
{
  __int16 v12; // bx
  _BYTE *v13; // rsi
  _BYTE *v14; // rdx
  __int64 v15; // rdi
  unsigned __int32 v16; // edx
  bool v17; // zf
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 result; // rax
  unsigned __int32 v24; // edx
  __int64 v25; // rdi
  unsigned __int32 v26; // edx
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rdi
  __int64 v30; // rdi
  unsigned int v31; // [rsp+4h] [rbp-4Ch]
  unsigned int v32; // [rsp+4h] [rbp-4Ch]
  unsigned int v33; // [rsp+4h] [rbp-4Ch]
  unsigned __int8 v36[56]; // [rsp+18h] [rbp-38h] BYREF

  v12 = a7;
  v13 = *(_BYTE **)(a1 + 24);
  v14 = *(_BYTE **)(a1 + 16);
  if ( a7 <= 0xFEFF || a8 == 1 )
  {
    if ( v13 != v14 )
    {
      *(_DWORD *)v36 = 0;
      if ( v13 == *(_BYTE **)(a1 + 32) )
      {
        v32 = a2;
        sub_C88AB0(a1 + 16, v13, v36);
        v12 = a7;
        a2 = v32;
      }
      else
      {
        if ( v13 )
        {
          *(_DWORD *)v13 = 0;
          v13 = *(_BYTE **)(a1 + 24);
          v12 = a7;
        }
        *(_QWORD *)(a1 + 24) = v13 + 4;
      }
    }
  }
  else if ( v13 == v14
         && (!*(_DWORD *)(a1 + 40)
          || (v31 = a2,
              sub_C17A60(a1 + 16, *(unsigned int *)(a1 + 40)),
              v13 = *(_BYTE **)(a1 + 24),
              a2 = v31,
              v13 == *(_BYTE **)(a1 + 16))) )
  {
    v12 = -1;
  }
  else if ( v13 == *(_BYTE **)(a1 + 32) )
  {
    v33 = a2;
    v12 = -1;
    sub_B8BBF0(a1 + 16, v13, &a7);
    a2 = v33;
  }
  else
  {
    if ( v13 )
      *(_DWORD *)v13 = a7;
    *(_QWORD *)(a1 + 24) += 4LL;
    v12 = -1;
  }
  v15 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  v16 = _byteswap_ulong(a2);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
    a2 = v16;
  v17 = *(_BYTE *)(a1 + 8) == 0;
  *(_DWORD *)v36 = a2;
  if ( v17 )
  {
    sub_CB6200(v15, v36, 4u);
    v24 = a4;
    v25 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
      v24 = _byteswap_ulong(a4);
    *(_DWORD *)v36 = v24;
    sub_CB6200(v25, v36, 4u);
    v26 = a5;
    v27 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
      v26 = _byteswap_ulong(a5);
    *(_DWORD *)v36 = v26;
    sub_CB6200(v27, v36, 4u);
    v28 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v36[0] = a3;
    sub_CB6200(v28, v36, 1u);
    v29 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v36[0] = a6;
    sub_CB6200(v29, v36, 1u);
    v30 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
      v12 = __ROL2__(v12, 8);
    *(_WORD *)v36 = v12;
    result = sub_CB6200(v30, v36, 2u);
  }
  else
  {
    sub_CB6200(v15, v36, 4u);
    v18 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v36[0] = a3;
    sub_CB6200(v18, v36, 1u);
    v19 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v36[0] = a6;
    sub_CB6200(v19, v36, 1u);
    v20 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
      v12 = __ROL2__(v12, 8);
    *(_WORD *)v36 = v12;
    sub_CB6200(v20, v36, 2u);
    v21 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
      a4 = _byteswap_uint64(a4);
    *(_QWORD *)v36 = a4;
    sub_CB6200(v21, v36, 8u);
    v22 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 16LL) != 1 )
      a5 = _byteswap_uint64(a5);
    *(_QWORD *)v36 = a5;
    result = sub_CB6200(v22, v36, 8u);
  }
  ++*(_DWORD *)(a1 + 40);
  return result;
}
