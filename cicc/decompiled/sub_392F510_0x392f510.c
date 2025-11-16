// Function: sub_392F510
// Address: 0x392f510
//
__int64 __fastcall sub_392F510(
        __int64 a1,
        unsigned int a2,
        char a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        char a6,
        unsigned int a7,
        char a8)
{
  __int16 v12; // r12
  _BYTE *v13; // rsi
  _BYTE *v14; // rdx
  unsigned __int32 v15; // esi
  __int64 v16; // rdi
  bool v17; // zf
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int16 v20; // dx
  __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // rdi
  unsigned __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 result; // rax
  __int64 v27; // rdi
  unsigned __int32 v28; // eax
  __int64 v29; // rdi
  unsigned __int32 v30; // eax
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int16 v33; // dx
  __int64 v34; // rdi
  unsigned int v35; // [rsp+4h] [rbp-4Ch]
  unsigned int v36; // [rsp+4h] [rbp-4Ch]
  unsigned int v37; // [rsp+4h] [rbp-4Ch]
  char v40[56]; // [rsp+18h] [rbp-38h] BYREF

  v12 = a7;
  v13 = *(_BYTE **)(a1 + 24);
  v14 = *(_BYTE **)(a1 + 16);
  if ( a7 <= 0xFEFF || a8 == 1 )
  {
    if ( v14 != v13 )
    {
      *(_DWORD *)v40 = 0;
      if ( *(_BYTE **)(a1 + 32) == v13 )
      {
        v36 = a2;
        sub_C88AB0(a1 + 16, v13, v40);
        v12 = a7;
        a2 = v36;
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
  else if ( v14 == v13
         && (!*(_DWORD *)(a1 + 40)
          || (v35 = a2,
              sub_C17A60(a1 + 16, *(unsigned int *)(a1 + 40)),
              v13 = *(_BYTE **)(a1 + 24),
              a2 = v35,
              v13 == *(_BYTE **)(a1 + 16))) )
  {
    v12 = -1;
  }
  else if ( v13 == *(_BYTE **)(a1 + 32) )
  {
    v37 = a2;
    v12 = -1;
    sub_B8BBF0(a1 + 16, v13, &a7);
    a2 = v37;
  }
  else
  {
    if ( v13 )
      *(_DWORD *)v13 = a7;
    *(_QWORD *)(a1 + 24) += 4LL;
    v12 = -1;
  }
  v15 = _byteswap_ulong(a2);
  v16 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) > 1 )
    a2 = v15;
  v17 = *(_BYTE *)(a1 + 8) == 0;
  *(_DWORD *)v40 = a2;
  if ( v17 )
  {
    sub_16E7EE0(v16, v40, 4u);
    v27 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v28 = _byteswap_ulong(a4);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) <= 1 )
      v28 = a4;
    *(_DWORD *)v40 = v28;
    sub_16E7EE0(v27, v40, 4u);
    v29 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v30 = _byteswap_ulong(a5);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) <= 1 )
      v30 = a5;
    *(_DWORD *)v40 = v30;
    sub_16E7EE0(v29, v40, 4u);
    v31 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v40[0] = a3;
    sub_16E7EE0(v31, v40, 1u);
    v32 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v40[0] = a6;
    sub_16E7EE0(v32, v40, 1u);
    v33 = __ROL2__(v12, 8);
    v34 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) > 1 )
      v12 = v33;
    *(_WORD *)v40 = v12;
    result = sub_16E7EE0(v34, v40, 2u);
  }
  else
  {
    sub_16E7EE0(v16, v40, 4u);
    v18 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v40[0] = a3;
    sub_16E7EE0(v18, v40, 1u);
    v19 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    v40[0] = a6;
    sub_16E7EE0(v19, v40, 1u);
    v20 = __ROL2__(v12, 8);
    v21 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) > 1 )
      v12 = v20;
    *(_WORD *)v40 = v12;
    sub_16E7EE0(v21, v40, 2u);
    v22 = _byteswap_uint64(a4);
    v23 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) > 1 )
      a4 = v22;
    *(_QWORD *)v40 = a4;
    sub_16E7EE0(v23, v40, 8u);
    v24 = _byteswap_uint64(a5);
    v25 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 16LL) - 1) > 1 )
      a5 = v24;
    *(_QWORD *)v40 = a5;
    result = sub_16E7EE0(v25, v40, 8u);
  }
  ++*(_DWORD *)(a1 + 40);
  return result;
}
