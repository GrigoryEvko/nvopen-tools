// Function: sub_1E85940
// Address: 0x1e85940
//
__int64 __fastcall sub_1E85940(__int64 a1, unsigned int a2)
{
  void *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rdx
  _BYTE *v5; // rax
  __int64 result; // rax
  _BYTE v7[16]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v8)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  void (__fastcall *v9)(_BYTE *, __int64); // [rsp+18h] [rbp-28h]

  v2 = sub_16E8CB0();
  v3 = sub_1263B40((__int64)v2, "- v. register: ");
  sub_1F4AA00(v7, a2, *(_QWORD *)(a1 + 40), 0, 0);
  if ( !v8 )
    sub_4263D6(v7, a2, v4);
  v9(v7, v3);
  v5 = *(_BYTE **)(v3 + 24);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(v3 + 16) )
  {
    sub_16E7DE0(v3, 10);
  }
  else
  {
    *(_QWORD *)(v3 + 24) = v5 + 1;
    *v5 = 10;
  }
  result = (__int64)v8;
  if ( v8 )
    return v8(v7, v7, 3);
  return result;
}
