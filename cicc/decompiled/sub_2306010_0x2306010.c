// Function: sub_2306010
// Address: 0x2306010
//
_QWORD *__fastcall sub_2306010(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r13
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  __int64 v6; // rdi
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_30B1B70(v8, a2 + 8);
  v3 = v8[0];
  v8[0] = 0;
  v4 = (_QWORD *)sub_22077B0(0x10u);
  v5 = v4;
  if ( v4 )
  {
    v4[1] = v3;
    *v4 = &unk_4A0ACA0;
  }
  else if ( v3 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  }
  v6 = v8[0];
  *a1 = v5;
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  return a1;
}
