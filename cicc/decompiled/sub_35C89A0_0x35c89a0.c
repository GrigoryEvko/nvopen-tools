// Function: sub_35C89A0
// Address: 0x35c89a0
//
__int64 __fastcall sub_35C89A0(__int64 a1, __int64 a2, void (*a3)(), __int64 a4)
{
  __int64 v5; // rdi
  void (*v6)(); // rax
  unsigned int v7; // r12d
  __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD v11[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( (*(_BYTE *)(a2 + 344) & 0x10) != 0 )
  {
    if ( *(_BYTE *)(a1 + 201) )
      sub_C64ED0("Instruction selection failed", 1u);
    sub_2E80380(a2, a2, a3, a4);
    sub_2E81690(a2);
    sub_2E78D90((_QWORD *)a2, *(_QWORD *)(a2 + 16));
    v5 = *(_QWORD *)(a2 + 8);
    v6 = *(void (**)())(*(_QWORD *)v5 + 248LL);
    if ( v6 != nullsub_1497 )
      ((void (__fastcall *)(__int64, __int64))v6)(v5, a2);
    v7 = *(unsigned __int8 *)(a1 + 200);
    if ( (_BYTE)v7 )
    {
      v9 = *(_QWORD *)a2;
      v11[1] = 0x10000000BLL;
      v11[2] = v9;
      v11[0] = &unk_49D9EE8;
      v10 = sub_B2BE50(v9);
      sub_B6EB20(v10, (__int64)v11);
    }
    else
    {
      v7 = 1;
    }
  }
  else
  {
    v7 = 0;
  }
  sub_2EBEA90(*(_QWORD *)(a2 + 32));
  return v7;
}
