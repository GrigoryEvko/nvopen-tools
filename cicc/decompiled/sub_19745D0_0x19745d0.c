// Function: sub_19745D0
// Address: 0x19745d0
//
__int64 __fastcall sub_19745D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // r8d
  int v13; // r9d
  __int64 result; // rax
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // rax
  __int64 v18; // [rsp+10h] [rbp-150h]
  unsigned __int8 v19; // [rsp+28h] [rbp-138h]
  unsigned __int8 v20; // [rsp+28h] [rbp-138h]
  unsigned __int8 v21; // [rsp+28h] [rbp-138h]
  _QWORD v22[2]; // [rsp+30h] [rbp-130h] BYREF
  __int64 v23; // [rsp+40h] [rbp-120h]
  int v24; // [rsp+48h] [rbp-118h]
  __int64 v25; // [rsp+50h] [rbp-110h]
  __int64 v26; // [rsp+58h] [rbp-108h]
  _BYTE *v27; // [rsp+60h] [rbp-100h]
  __int64 v28; // [rsp+68h] [rbp-F8h]
  _BYTE v29[16]; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v30[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+90h] [rbp-D0h]
  __int64 v32; // [rsp+98h] [rbp-C8h]
  __int64 v33; // [rsp+A0h] [rbp-C0h]
  __int64 v34; // [rsp+A8h] [rbp-B8h]
  __int64 v35; // [rsp+B0h] [rbp-B0h]
  char v36; // [rsp+B8h] [rbp-A8h]
  __int64 v37; // [rsp+C0h] [rbp-A0h]
  _BYTE *v38; // [rsp+C8h] [rbp-98h]
  _BYTE *v39; // [rsp+D0h] [rbp-90h]
  __int64 v40; // [rsp+D8h] [rbp-88h]
  int v41; // [rsp+E0h] [rbp-80h]
  _BYTE v42[120]; // [rsp+E8h] [rbp-78h] BYREF

  v6 = sub_157F280(**(_QWORD **)(a2 + 32));
  v18 = v7;
  if ( v6 == v7 )
    return 1;
  v8 = v6;
  while ( 1 )
  {
    v30[0] = 6;
    v38 = v42;
    v39 = v42;
    v30[1] = 0;
    v27 = v29;
    v28 = 0x200000000LL;
    v31 = 0;
    v11 = *(_QWORD *)(a1 + 16);
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v40 = 8;
    v41 = 0;
    v22[0] = 6;
    v22[1] = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    if ( (unsigned __int8)sub_1B16990(v8, a2, v11, v22, 0, 0) )
    {
      v9 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v9 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v12, v13);
        v9 = *(unsigned int *)(a3 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v8;
      ++*(_DWORD *)(a3 + 8);
      goto LABEL_6;
    }
    result = sub_1B1CF40(v8, a2, v30, 0, 0, 0);
    if ( !(_BYTE)result )
      break;
    v17 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v17 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v15, v16);
      v17 = *(unsigned int *)(a4 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * v17) = v8;
    ++*(_DWORD *)(a4 + 8);
LABEL_6:
    if ( v27 != v29 )
      _libc_free((unsigned __int64)v27);
    if ( v23 != 0 && v23 != -8 && v23 != -16 )
      sub_1649B30(v22);
    if ( v39 != v38 )
      _libc_free((unsigned __int64)v39);
    if ( v31 != 0 && v31 != -8 && v31 != -16 )
      sub_1649B30(v30);
    if ( !v8 )
      BUG();
    v10 = *(_QWORD *)(v8 + 32);
    if ( !v10 )
      BUG();
    v8 = 0;
    if ( *(_BYTE *)(v10 - 8) == 77 )
      v8 = v10 - 24;
    if ( v18 == v8 )
      return 1;
  }
  if ( v27 != v29 )
  {
    _libc_free((unsigned __int64)v27);
    result = 0;
  }
  if ( v23 != 0 && v23 != -8 && v23 != -16 )
  {
    v19 = result;
    sub_1649B30(v22);
    result = v19;
  }
  if ( v39 != v38 )
  {
    v20 = result;
    _libc_free((unsigned __int64)v39);
    result = v20;
  }
  if ( v31 != -8 && v31 != 0 && v31 != -16 )
  {
    v21 = result;
    sub_1649B30(v30);
    return v21;
  }
  return result;
}
