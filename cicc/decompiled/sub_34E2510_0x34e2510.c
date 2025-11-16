// Function: sub_34E2510
// Address: 0x34e2510
//
__int64 __fastcall sub_34E2510(__int64 a1, __int64 a2, size_t a3, __int64 *a4)
{
  __int64 result; // rax
  const char *v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int8 *v10; // r10
  const char *v11; // r8
  size_t v12; // r15
  unsigned __int64 v13; // rax
  _QWORD *v14; // rdx
  unsigned __int8 *v15; // r8
  size_t v16; // r9
  __int64 v17; // rax
  _QWORD *v18; // rdi
  const char *src; // [rsp+0h] [rbp-70h]
  unsigned __int8 *v20; // [rsp+8h] [rbp-68h]
  unsigned __int64 v21; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v22; // [rsp+20h] [rbp-50h] BYREF
  size_t v23; // [rsp+28h] [rbp-48h]
  _QWORD v24[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( a3 == 23 )
  {
    result = *(_QWORD *)a2 ^ 0x6156206775626544LL;
    if ( !(result | *(_QWORD *)(a2 + 8) ^ 0x4120656C62616972LL)
      && *(_DWORD *)(a2 + 16) == 2037145966
      && *(_WORD *)(a2 + 20) == 26995
      && *(_BYTE *)(a2 + 22) == 115 )
    {
      return result;
    }
  }
  sub_34E2200(a1, a4, 0);
  v8 = sub_2E791E0(a4);
  v10 = (unsigned __int8 *)a2;
  v11 = v8;
  v12 = v9;
  if ( !v8 )
  {
    v23 = 0;
    v16 = 0;
    v22 = v24;
    v15 = (unsigned __int8 *)v24;
    LOBYTE(v24[0]) = 0;
    goto LABEL_9;
  }
  v21 = v9;
  v13 = v9;
  v22 = v24;
  if ( v9 > 0xF )
  {
    src = v11;
    v17 = sub_22409D0((__int64)&v22, &v21, 0);
    v10 = (unsigned __int8 *)a2;
    v11 = src;
    v22 = (_QWORD *)v17;
    v18 = (_QWORD *)v17;
    v24[0] = v21;
LABEL_17:
    v20 = v10;
    memcpy(v18, v11, v12);
    v13 = v21;
    v14 = v22;
    v10 = v20;
    goto LABEL_7;
  }
  if ( v9 != 1 )
  {
    if ( !v9 )
    {
      v14 = v24;
      goto LABEL_7;
    }
    v18 = v24;
    goto LABEL_17;
  }
  LOBYTE(v24[0]) = *v11;
  v14 = v24;
LABEL_7:
  v23 = v13;
  *((_BYTE *)v14 + v13) = 0;
  v15 = (unsigned __int8 *)v22;
  v16 = v23;
LABEL_9:
  sub_34E1EC0(a1, a4, v10, a3, v15, v16);
  if ( v22 != v24 )
    j_j___libc_free_0((unsigned __int64)v22);
  return sub_3140C00(a1);
}
