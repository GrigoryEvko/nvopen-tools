// Function: sub_23A23F0
// Address: 0x23a23f0
//
unsigned __int64 __fastcall sub_23A23F0(unsigned __int64 *a1, char *a2)
{
  char v2; // r14
  __int64 v3; // r13
  __int64 v4; // r12
  int v5; // ebx
  __int64 v6; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *a2;
  v3 = *((_QWORD *)a2 + 1);
  v4 = *((_QWORD *)a2 + 2);
  v5 = *((_DWORD *)a2 + 6);
  v6 = sub_22077B0(0x28u);
  if ( v6 )
  {
    *(_BYTE *)(v6 + 8) = v2;
    *(_QWORD *)(v6 + 16) = v3;
    *(_QWORD *)(v6 + 24) = v4;
    *(_QWORD *)v6 = &unk_4A0D7B8;
    *(_DWORD *)(v6 + 32) = v5;
  }
  v8[0] = v6;
  result = sub_23A2230(a1, v8);
  if ( v8[0] )
    return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v8[0] + 8LL))(v8[0]);
  return result;
}
