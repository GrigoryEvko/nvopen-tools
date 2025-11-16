// Function: sub_EB9E70
// Address: 0xeb9e70
//
__int64 __fastcall sub_EB9E70(__int64 a1)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r12d
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  void (*v8)(); // rdx
  const char *v9; // [rsp+0h] [rbp-60h] BYREF
  const char *v10; // [rsp+8h] [rbp-58h]
  const char *v11[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v12; // [rsp+30h] [rbp-30h]

  v9 = 0;
  v10 = 0;
  v11[0] = "expected identifier";
  v12 = 259;
  v2 = sub_EB61F0(a1, (__int64 *)&v9);
  if ( (unsigned __int8)sub_ECE0A0(a1, v2, v11) )
    return 1;
  v3 = sub_ECE000(a1);
  if ( (_BYTE)v3 )
  {
    return 1;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 224);
    v12 = 261;
    v11[0] = v9;
    v11[1] = v10;
    v6 = sub_E6C460(v5, v11);
    v7 = *(_QWORD *)(a1 + 232);
    v8 = *(void (**)())(*(_QWORD *)v7 + 1216LL);
    if ( v8 != nullsub_114 )
      ((void (__fastcall *)(__int64, __int64))v8)(v7, v6);
  }
  return v3;
}
