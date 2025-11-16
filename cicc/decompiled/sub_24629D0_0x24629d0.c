// Function: sub_24629D0
// Address: 0x24629d0
//
__int64 __fastcall sub_24629D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  __int64 *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v12; // [rsp+8h] [rbp-58h]
  _QWORD v13[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 48) == 33 )
  {
    v12 = *(_QWORD *)(a1 + 96);
    v9 = (__int64 *)sub_BCB120(*(_QWORD **)(a1 + 72));
    v13[1] = a5;
    v13[0] = v12;
    v8 = 2;
  }
  else
  {
    v13[0] = a5;
    v8 = 1;
    v9 = *(__int64 **)(a1 + 496);
  }
  v10 = sub_BCF480(v9, v13, v8, 0);
  return sub_BA8C10(a2, a3, a4, v10, 0);
}
