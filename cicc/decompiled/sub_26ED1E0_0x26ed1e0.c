// Function: sub_26ED1E0
// Address: 0x26ed1e0
//
void __fastcall sub_26ED1E0(__int64 *a1, _BYTE *a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdi
  _QWORD v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *a4;
  *a4 = 0;
  v6 = *a1;
  if ( v5 )
  {
    (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)v5 + 16LL))(v8, v5);
    sub_26ECF60(v6, a2, a3, v8);
    if ( v8[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v8[0] + 8LL))(v8[0]);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  }
  else
  {
    v7 = *a1;
    v8[0] = 0;
    sub_26ECF60(v7, a2, a3, v8);
    if ( v8[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v8[0] + 8LL))(v8[0]);
  }
}
