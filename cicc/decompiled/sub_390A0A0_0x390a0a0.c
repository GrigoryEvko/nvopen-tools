// Function: sub_390A0A0
// Address: 0x390a0a0
//
__int64 __fastcall sub_390A0A0(__int64 a1, _DWORD *a2, __int64 a3)
{
  unsigned int v5; // eax
  __int64 v6; // rax
  _BOOL8 v7; // rcx
  __int64 v8; // rdi
  __int64 v10; // rax
  _BOOL8 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  (*(void (__fastcall **)(__int64 *))(*(_QWORD *)a2 + 24LL))(&v14);
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 16LL))(v14);
  if ( v5 == 3 )
  {
    v10 = v14;
    v11 = a2[4] == 1;
    v14 = 0;
    v15[0] = v10;
    sub_3915AB0(a1, v15, a3, v11);
    v8 = v15[0];
    if ( !v15[0] )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( v5 > 3 )
  {
    v13 = v14;
    v14 = 0;
    v15[0] = v13;
    sub_391D7F0(a1, v15, a3);
    v8 = v15[0];
    if ( !v15[0] )
      goto LABEL_6;
LABEL_5:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
    goto LABEL_6;
  }
  if ( v5 != 1 )
  {
    v6 = v14;
    v7 = a2[4] == 1;
    v14 = 0;
    v15[0] = v6;
    sub_392EBD0(a1, v15, a3, v7);
    v8 = v15[0];
    if ( !v15[0] )
      goto LABEL_6;
    goto LABEL_5;
  }
  v12 = v14;
  v14 = 0;
  v15[0] = v12;
  sub_392A120(a1, v15, a3);
  v8 = v15[0];
  if ( v15[0] )
    goto LABEL_5;
LABEL_6:
  if ( v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  return a1;
}
