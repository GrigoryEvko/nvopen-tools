// Function: sub_30FD0F0
// Address: 0x30fd0f0
//
__int64 *__fastcall sub_30FD0F0(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  char v9; // [rsp+Ch] [rbp-34h]

  v9 = *(_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 80) + 24LL))(*(_QWORD *)(a2 + 80)) != 0;
  v6 = sub_22077B0(0x228u);
  v7 = v6;
  if ( v6 )
    sub_30FCF60(v6, a2, a3, a4, v9);
  *a1 = v7;
  return a1;
}
