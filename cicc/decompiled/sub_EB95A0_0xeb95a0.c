// Function: sub_EB95A0
// Address: 0xeb95a0
//
__int64 __fastcall sub_EB95A0(__int64 a1)
{
  __int64 v2; // r14
  unsigned int v3; // r13d
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  void (*v8)(); // rcx
  const char *v9; // [rsp+0h] [rbp-60h] BYREF
  const char *v10; // [rsp+8h] [rbp-58h]
  const char *v11[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v12; // [rsp+30h] [rbp-30h]

  v9 = 0;
  v10 = 0;
  v2 = sub_ECD690(a1 + 40);
  if ( (unsigned __int8)sub_EB61F0(a1, (__int64 *)&v9) )
  {
    v11[0] = "expected symbol name";
    v12 = 259;
    return (unsigned int)sub_ECE0E0(a1, v11, 0, 0);
  }
  else
  {
    v3 = sub_ECE000(a1);
    if ( !(_BYTE)v3 )
    {
      v5 = *(_QWORD *)(a1 + 224);
      v12 = 261;
      v11[0] = v9;
      v11[1] = v10;
      v6 = sub_E6C460(v5, v11);
      v7 = *(_QWORD *)(a1 + 232);
      v8 = *(void (**)())(*(_QWORD *)v7 + 824LL);
      if ( v8 != nullsub_112 )
        ((void (__fastcall *)(__int64, __int64, __int64))v8)(v7, v6, v2);
    }
  }
  return v3;
}
