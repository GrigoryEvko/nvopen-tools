// Function: sub_81B840
// Address: 0x81b840
//
_BYTE *sub_81B840()
{
  _BYTE *v0; // rbx
  __int64 v1; // rax
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rax
  _BYTE *result; // rax
  __int64 v6; // rax
  __int64 v7; // [rsp+0h] [rbp-20h] BYREF
  __int64 v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v0 = qword_4F06460 + 2;
  v1 = sub_7AF1D0((unsigned __int64)(qword_4F06460 + 2));
  v2 = (unsigned __int64)&v0[*(_QWORD *)(v1 + 32)];
  v7 = v1;
  v3 = sub_7AF1D0(v2);
  v8[0] = v3;
  if ( dword_4F19500 )
  {
    sub_7AEF90(v7);
    sub_7AEF30((__int64)&v7);
    v4 = v8[0];
    *(_BYTE *)(v8[0] + 48) &= ~0x80u;
    sub_81B790(*(_QWORD *)(v4 + 16), *(_QWORD *)(v4 + 32));
  }
  else
  {
    sub_7AEF90(v3);
    sub_7AEF30((__int64)v8);
    v6 = v7;
    *(_BYTE *)(v7 + 48) &= ~0x80u;
    sub_81B790(*(_QWORD *)(v6 + 16), *(_QWORD *)(v6 + 32));
  }
  result = qword_4F06460;
  qword_4F06460[1] = 4;
  return result;
}
