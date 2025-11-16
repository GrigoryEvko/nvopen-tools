// Function: sub_23A1F80
// Address: 0x23a1f80
//
unsigned __int64 __fastcall sub_23A1F80(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r13
  _QWORD *v5; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a2[2];
  v3 = *a2;
  v4 = a2[1];
  v5 = (_QWORD *)sub_22077B0(0x20u);
  if ( v5 )
  {
    v5[1] = v3;
    v5[2] = v4;
    v5[3] = v2;
    *v5 = &unk_4A11BB8;
  }
  v7[0] = (unsigned __int64)v5;
  result = sub_23A1F40(a1, v7);
  if ( v7[0] )
    return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v7[0] + 8LL))(v7[0]);
  return result;
}
