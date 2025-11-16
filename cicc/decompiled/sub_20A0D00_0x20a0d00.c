// Function: sub_20A0D00
// Address: 0x20a0d00
//
unsigned __int64 __fastcall sub_20A0D00(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r15
  int v6; // eax
  __int64 v7; // r15
  __int64 v8; // r12
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r14
  unsigned __int64 result; // rax
  unsigned __int64 v15; // r12
  size_t v16; // rax
  __int64 v17; // rdx
  unsigned int v21; // [rsp+18h] [rbp-68h]
  int v22; // [rsp+1Ch] [rbp-64h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  int v24; // [rsp+28h] [rbp-58h]
  __int64 v25; // [rsp+30h] [rbp-50h] BYREF
  __int64 v26; // [rsp+38h] [rbp-48h]
  __int64 v27; // [rsp+40h] [rbp-40h]

  v5 = a1;
  v6 = *(_DWORD *)(a2 + 24);
  if ( v6 == 1 )
  {
    v13 = a2 + 192;
    sub_2240AE0(a2 + 192, *(_QWORD *)(a2 + 16));
    *(_DWORD *)(a2 + 224) = (*(__int64 (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 1384))(
                              a1,
                              *(_QWORD *)(a2 + 192),
                              *(_QWORD *)(a2 + 200));
    goto LABEL_16;
  }
  if ( !v6 )
  {
    v22 = 4;
    v8 = 0;
    goto LABEL_15;
  }
  v23 = *(unsigned int *)(a2 + 24);
  v7 = 0;
  v22 = 4;
  v21 = 0;
  v24 = -1;
  while ( 1 )
  {
    v8 = 32 * v7;
    v9 = (*(__int64 (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 1384))(
           a1,
           *(_QWORD *)(32 * v7 + *(_QWORD *)(a2 + 16)),
           *(_QWORD *)(32 * v7 + *(_QWORD *)(a2 + 16) + 8));
    v10 = v9;
    if ( v9 == 3 )
      break;
    if ( v9 != 2 || *(_DWORD *)(a2 + 4) == -1 )
      goto LABEL_6;
LABEL_8:
    if ( v23 == ++v7 )
    {
      v5 = a1;
      v8 = 32LL * v21;
      goto LABEL_15;
    }
  }
  if ( !a3 )
  {
LABEL_6:
    if ( dword_430A260[v10] > v24 )
    {
      v24 = dword_430A260[v10];
      v22 = v10;
      v21 = v7;
    }
    goto LABEL_8;
  }
  v11 = *(_QWORD *)(a2 + 16);
  v12 = *a1;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64 *, __int64))(v12 + 1416))(
    a1,
    a3,
    a4,
    v8 + v11,
    &v25,
    a5);
  if ( v26 == v25 )
  {
    if ( v25 )
      j_j___libc_free_0(v25, v27 - v25);
    goto LABEL_6;
  }
  v5 = a1;
  if ( v25 )
    j_j___libc_free_0(v25, v27 - v25);
  v22 = 3;
LABEL_15:
  v13 = a2 + 192;
  sub_2240AE0(a2 + 192, v8 + *(_QWORD *)(a2 + 16));
  *(_DWORD *)(a2 + 224) = v22;
LABEL_16:
  result = sub_2241AC0(v13, "X");
  if ( !(_DWORD)result )
  {
    result = *(_QWORD *)(a2 + 232);
    if ( result )
    {
      result = *(unsigned __int8 *)(result + 16);
      if ( (unsigned __int8)result > 0x12u || (v17 = 270337, !_bittest64(&v17, result)) )
      {
        result = (*(__int64 (__fastcall **)(__int64 *, _QWORD, _QWORD))(*v5 + 1408))(
                   v5,
                   *(unsigned __int8 *)(a2 + 240),
                   0);
        v15 = result;
        if ( result )
        {
          v16 = strlen((const char *)result);
          sub_2241130(v13, 0, *(_QWORD *)(a2 + 200), v15, v16);
          result = (*(__int64 (__fastcall **)(__int64 *, _QWORD, _QWORD))(*v5 + 1384))(
                     v5,
                     *(_QWORD *)(a2 + 192),
                     *(_QWORD *)(a2 + 200));
          *(_DWORD *)(a2 + 224) = result;
        }
      }
    }
  }
  return result;
}
