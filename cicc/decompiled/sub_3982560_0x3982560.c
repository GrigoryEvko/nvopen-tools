// Function: sub_3982560
// Address: 0x3982560
//
__int64 __fastcall sub_3982560(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rdi
  __int64 v5; // rax
  void (*v6)(); // rcx
  __int64 v7; // rax
  __int64 v8; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v9[2]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v10; // [rsp+20h] [rbp-60h]
  _QWORD v11[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v12; // [rsp+40h] [rbp-40h]
  _QWORD v13[2]; // [rsp+50h] [rbp-30h] BYREF
  __int16 v14; // [rsp+60h] [rbp-20h]

  v3 = *(__int64 **)(a2 + 256);
  if ( *(_BYTE *)(a2 + 416) )
  {
    v5 = *v3;
    v9[0] = a1;
    v12 = 2818;
    v6 = *(void (**)())(v5 + 104);
    v7 = a1[1];
    v14 = 770;
    v8 = v7 + 1;
    v10 = 773;
    v9[1] = " [";
    v11[0] = v9;
    v11[1] = &v8;
    v13[0] = v11;
    v13[1] = " bytes]";
    if ( v6 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v6)(v3, v13, 1);
      v3 = *(__int64 **)(a2 + 256);
    }
  }
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*v3 + 400))(v3, *a1, a1[1]);
  return sub_396F300(a2, 0);
}
