// Function: sub_121D100
// Address: 0x121d100
//
__int64 __fastcall sub_121D100(__int64 a1, _BYTE *a2, size_t a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v5; // r15
  int v9; // eax
  __int64 v10; // r8
  size_t v11; // rax
  const char *v12; // rax
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  int v17; // eax
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // [rsp-80h] [rbp-80h]
  __int64 v21; // [rsp-80h] [rbp-80h]
  size_t v22; // [rsp-70h] [rbp-70h] BYREF
  _QWORD v23[2]; // [rsp-68h] [rbp-68h] BYREF
  _QWORD v24[2]; // [rsp-58h] [rbp-58h] BYREF
  char v25; // [rsp-48h] [rbp-48h]
  char v26; // [rsp-47h] [rbp-47h]

  *a4 = 0;
  result = 0;
  if ( *(_DWORD *)(a1 + 240) != 294 )
    return result;
  v5 = a1 + 176;
  v20 = *(_QWORD *)(a1 + 232);
  v9 = sub_1205200(a1 + 176);
  v10 = v20;
  *(_DWORD *)(a1 + 240) = v9;
  if ( v9 == 12 )
  {
    v17 = sub_1205200(v5);
    *(_DWORD *)(a1 + 240) = v17;
    if ( v17 == 509 )
    {
      *a4 = sub_121CE20(a1, a1 + 248, *(_QWORD *)(a1 + 232));
      *(_DWORD *)(a1 + 240) = sub_1205200(v5);
      return sub_120AFE0(a1, 13, "expected ')' after comdat var");
    }
    v26 = 1;
    v12 = "expected comdat variable";
    goto LABEL_5;
  }
  v11 = a3;
  if ( !a3 )
  {
    v26 = 1;
    v12 = "comdat cannot be unnamed";
LABEL_5:
    v23[0] = v12;
    v13 = *(_QWORD *)(a1 + 232);
    v25 = 3;
    sub_11FD800(v5, v13, (__int64)v23, 1);
    return 1;
  }
  v23[0] = v24;
  if ( !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v22 = a3;
  if ( a3 > 0xF )
  {
    v19 = sub_22409D0(v23, &v22, 0);
    v10 = v20;
    v23[0] = v19;
    v18 = (_QWORD *)v19;
    v24[0] = v22;
    goto LABEL_17;
  }
  if ( a3 != 1 )
  {
    v18 = v24;
LABEL_17:
    v21 = v10;
    memcpy(v18, a2, a3);
    v11 = v22;
    v14 = (_QWORD *)v23[0];
    v10 = v21;
    goto LABEL_11;
  }
  LOBYTE(v24[0]) = *a2;
  v14 = v24;
LABEL_11:
  v23[1] = v11;
  *((_BYTE *)v14 + v11) = 0;
  v15 = sub_121CE20(a1, (__int64)v23, v10);
  v16 = (_QWORD *)v23[0];
  *a4 = v15;
  if ( v16 != v24 )
    j_j___libc_free_0(v16, v24[0] + 1LL);
  return 0;
}
