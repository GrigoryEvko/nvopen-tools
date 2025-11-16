// Function: sub_38EC8A0
// Address: 0x38ec8a0
//
__int64 __fastcall sub_38EC8A0(__int64 *a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 v8; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a1;
  v9[0] = 0;
  LOBYTE(v3) = sub_38EB6A0(v2, &v8, (__int64)v9);
  v4 = v3;
  if ( (_BYTE)v3 )
    return v4;
  v5 = *(__int64 **)(*a1 + 328);
  v6 = *v5;
  if ( *(_BYTE *)a1[1] )
  {
    (*(void (__fastcall **)(__int64 *, __int64))(v6 + 440))(v5, v8);
    return v4;
  }
  (*(void (__fastcall **)(__int64 *, __int64))(v6 + 432))(v5, v8);
  return v4;
}
