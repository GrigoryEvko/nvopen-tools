// Function: sub_B7F530
// Address: 0xb7f530
//
__int64 *__fastcall sub_B7F530(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v5; // rsi
  _BYTE v6[32]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v7[4]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v8; // [rsp+40h] [rbp-20h]

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v5 = *a2;
    *a2 = 0;
    (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v5 + 24LL))(v6);
    v7[2] = v6;
    v8 = 1027;
    v7[0] = "Error reading bitcode file: ";
    sub_C64D30(v7, 1);
  }
  v3 = *a2;
  *a2 = 0;
  *a1 = v3 | 1;
  return a1;
}
