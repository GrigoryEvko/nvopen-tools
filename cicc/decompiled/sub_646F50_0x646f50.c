// Function: sub_646F50
// Address: 0x646f50
//
__int64 __fastcall sub_646F50(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r12
  _DWORD v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  sub_7296C0(v8);
  v5 = sub_725FD0(v8, a2, v4);
  *(_QWORD *)(v5 + 152) = a1;
  v6 = v5;
  *(_BYTE *)(v5 + 172) = a2;
  if ( a3 != -1 )
    sub_7362F0(v5, a3);
  sub_729730(v8[0]);
  return v6;
}
