// Function: sub_321ADB0
// Address: 0x321adb0
//
void __fastcall sub_321ADB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdi
  __int64 (__fastcall *v5)(__int64, unsigned int); // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 *v8; // rdi
  __int64 v9; // rax
  void (__fastcall *v10)(__int64 *, __int64, unsigned __int64 **, _QWORD); // rax
  unsigned __int64 v11[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v12[16]; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int64 *v13; // [rsp+20h] [rbp-50h] BYREF
  __int16 v14; // [rsp+40h] [rbp-30h]

  v3 = *(_QWORD *)(a1 + 16);
  v11[0] = (unsigned __int64)v12;
  v11[1] = 0;
  v4 = *(_QWORD **)(v3 + 184);
  v12[0] = 0;
  v5 = *(__int64 (__fastcall **)(__int64, unsigned int))(*v4 + 480LL);
  if ( v5 == sub_31D48B0 )
  {
    v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v4[29] + 16LL) + 200LL))(*(_QWORD *)(v4[29] + 16LL));
    v7 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v6 + 16LL))(v6, (unsigned int)a2, 0);
  }
  else
  {
    v7 = ((__int64 (__fastcall *)(_QWORD *, __int64, unsigned __int64 *))v5)(v4, a2, v11);
  }
  if ( *(_BYTE *)(a1 + 120) )
    v8 = (__int64 *)(*(_QWORD *)(a1 + 104) + 80LL);
  else
    v8 = *(__int64 **)(a1 + 112);
  v9 = *v8;
  v13 = v11;
  v10 = *(void (__fastcall **)(__int64 *, __int64, unsigned __int64 **, _QWORD))(v9 + 16);
  v14 = 260;
  v10(v8, v7, &v13, 0);
  if ( (_BYTE *)v11[0] != v12 )
    j_j___libc_free_0(v11[0]);
}
