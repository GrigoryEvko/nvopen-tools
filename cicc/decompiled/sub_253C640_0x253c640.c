// Function: sub_253C640
// Address: 0x253c640
//
__int64 *__fastcall sub_253C640(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax
  _QWORD v6[12]; // [rsp+0h] [rbp-60h] BYREF

  sub_253C590(a1, "nofpclass");
  memset(&v6[1], 0, 32);
  v6[5] = 0x100000000LL;
  v6[0] = &unk_49DD210;
  v6[6] = a1;
  sub_CB5980((__int64)v6, 0, 0, 0);
  v3 = sub_C65140((__int64)v6, *(_DWORD *)(a2 + 96));
  v4 = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 24) )
  {
    v3 = sub_CB5D20(v3, 47);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = v4 + 1;
    *v4 = 47;
  }
  sub_C65140(v3, *(_DWORD *)(a2 + 100));
  v6[0] = &unk_49DD210;
  sub_CB5840((__int64)v6);
  return a1;
}
