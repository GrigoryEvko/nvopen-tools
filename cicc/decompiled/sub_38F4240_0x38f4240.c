// Function: sub_38F4240
// Address: 0x38f4240
//
__int64 __fastcall sub_38F4240(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rdx
  unsigned int v5; // ecx
  char v6; // al
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r13d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  void (*v14)(); // rcx
  __int64 v15[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v17; // [rsp+20h] [rbp-40h]

  v2 = sub_3909290(a1 + 144);
  v15[0] = 0;
  v15[1] = 0;
  v3 = v2;
  v6 = sub_38F0EE0(a1, v15, v4, v5);
  HIBYTE(v17) = 1;
  if ( v6 )
  {
    v16[0] = "expected symbol name";
    LOBYTE(v17) = 3;
    return (unsigned int)sub_3909CF0(a1, v16, 0, 0, v7, v8);
  }
  else
  {
    LOBYTE(v17) = 3;
    v16[0] = "unexpected tokens";
    v9 = sub_3909DC0(a1, v16);
    if ( (_BYTE)v9 )
    {
      v16[0] = " in '.cv_fpo_data' directive";
      v17 = 259;
      return (unsigned int)sub_39094A0(a1, v16);
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 320);
      v16[0] = v15;
      v17 = 261;
      v12 = sub_38BF510(v11, (__int64)v16);
      v13 = *(_QWORD *)(a1 + 328);
      v14 = *(void (**)())(*(_QWORD *)v13 + 680LL);
      if ( v14 != nullsub_594 )
        ((void (__fastcall *)(__int64, __int64, __int64))v14)(v13, v12, v3);
    }
  }
  return v9;
}
