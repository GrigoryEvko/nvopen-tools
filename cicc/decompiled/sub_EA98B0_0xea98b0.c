// Function: sub_EA98B0
// Address: 0xea98b0
//
__int64 __fastcall sub_EA98B0(__int64 a1, const char *a2, size_t a3, int a4)
{
  char v5; // bl
  __int64 v6; // rax
  unsigned int v7; // r14d
  bool v8; // zf
  __int64 v9; // rbx
  __int64 v10; // rcx
  int v11; // eax
  __int64 v13; // [rsp+0h] [rbp-90h]
  __int64 v14; // [rsp+8h] [rbp-88h]
  __int64 v16; // [rsp+18h] [rbp-78h]
  __int64 v17; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v18; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v19[4]; // [rsp+30h] [rbp-60h] BYREF
  char v20; // [rsp+50h] [rbp-40h]
  char v21; // [rsp+51h] [rbp-3Fh]

  v5 = a4;
  v6 = sub_ECD7B0(a1);
  v14 = sub_ECD6A0(v6);
  v7 = sub_EA83C0(a2, a3, v5 & 0xFD ^ 1u, (_QWORD *)a1, &v17, (__int64 *)&v18);
  if ( (_BYTE)v7 )
    return v7;
  v13 = v17;
  if ( !v17 )
    return v7;
  v8 = *(_QWORD *)(a1 + 856) == 0;
  v19[0] = a2;
  v19[1] = a3;
  if ( !v8 )
  {
    if ( a1 + 824 != sub_EA96F0(a1 + 816, (__int64)v19) )
      return v7;
    goto LABEL_13;
  }
  v9 = *(_QWORD *)(a1 + 768);
  v10 = v9 + 16LL * *(unsigned int *)(a1 + 776);
  if ( v9 == v10 )
    goto LABEL_13;
  while ( 1 )
  {
    if ( *(_QWORD *)(v9 + 8) == a3 )
    {
      if ( !a3 )
        break;
      v16 = v10;
      v11 = memcmp(*(const void **)v9, a2, a3);
      v10 = v16;
      if ( !v11 )
        break;
    }
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_13;
  }
  if ( v10 == v9 )
  {
LABEL_13:
    if ( a4 == 2 )
    {
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *))(**(_QWORD **)(a1 + 232) + 272LL))(
        *(_QWORD *)(a1 + 232),
        v13,
        v18);
    }
    else if ( a4 == 3 )
    {
      if ( *v18 == 2 )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *))(**(_QWORD **)(a1 + 232) + 280LL))(
          *(_QWORD *)(a1 + 232),
          v13,
          v18);
      }
      else
      {
        v21 = 1;
        v19[0] = "expected identifier";
        v20 = 3;
        return (unsigned int)sub_ECDA70(a1, v14, v19, 0, 0);
      }
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *))(**(_QWORD **)(a1 + 232) + 272LL))(
        *(_QWORD *)(a1 + 232),
        v13,
        v18);
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 232) + 296LL))(
        *(_QWORD *)(a1 + 232),
        v17,
        18);
    }
  }
  return v7;
}
