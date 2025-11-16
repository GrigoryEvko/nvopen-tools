// Function: sub_6EAEF0
// Address: 0x6eaef0
//
__int64 __fastcall sub_6EAEF0(__int64 a1, _QWORD *a2, _QWORD *a3, __int64 a4)
{
  __int64 v7; // rbx
  char v8; // dl
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(_BYTE *)(v7 + 80);
  v9 = v7;
  if ( v8 == 16 )
  {
    v9 = **(_QWORD **)(v7 + 88);
    v8 = *(_BYTE *)(v9 + 80);
  }
  if ( v8 == 24 )
    v9 = *(_QWORD *)(v9 + 88);
  v13 = *(_QWORD *)(v9 + 88);
  v10 = (__int64 *)sub_726700(4);
  v11 = *(_QWORD *)(v13 + 120);
  v10[7] = v13;
  *v10 = v11;
  sub_6E70E0(v10, a4);
  *(_QWORD *)(a4 + 136) = v7;
  *(_QWORD *)(a4 + 68) = *a2;
  *(_QWORD *)(a4 + 76) = *a3;
  sub_6E3280(a4, 0);
  *(_BYTE *)(a4 + 17) = 0;
  return sub_6E46C0(a4, a1);
}
