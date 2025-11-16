// Function: sub_1DB5300
// Address: 0x1db5300
//
__int64 __fastcall sub_1DB5300(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r12
  _WORD *v7; // rdx
  _BYTE *v8; // rax
  __int64 result; // rax
  int v10[4]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(int *, int *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v12)(int *, __int64, __int64, __int64, __int64, int); // [rsp+18h] [rbp-28h]

  v6 = a2;
  v7 = *(_WORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v7 <= 1u )
  {
    v6 = sub_16E7EE0(a2, " L", 2u);
  }
  else
  {
    *v7 = 19488;
    *(_QWORD *)(a2 + 24) += 2LL;
  }
  v10[0] = *(_DWORD *)(a1 + 112);
  v11 = (__int64 (__fastcall *)(int *, int *, int))sub_1DB3470;
  v12 = sub_1DB3430;
  sub_1DB3430(v10, v6, (__int64)v7, a4, a5, a6);
  v8 = *(_BYTE **)(v6 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v6 + 16) )
  {
    v6 = sub_16E7DE0(v6, 32);
  }
  else
  {
    *(_QWORD *)(v6 + 24) = v8 + 1;
    *v8 = 32;
  }
  sub_1DB50D0(a1, v6);
  result = (__int64)v11;
  if ( v11 )
    return v11(v10, v10, 3);
  return result;
}
