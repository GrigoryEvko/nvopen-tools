// Function: sub_302DC20
// Address: 0x302dc20
//
__int64 __fastcall sub_302DC20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v7; // rdi
  void (*v8)(); // r9
  __int64 *v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  unsigned __int8 v13; // [rsp+8h] [rbp-58h]
  unsigned int v14; // [rsp+Ch] [rbp-54h]
  _QWORD v15[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+30h] [rbp-30h]

  if ( *(_BYTE *)(a1 + 488) )
  {
    v7 = *(_QWORD *)(a1 + 224);
    v8 = *(void (**)())(*(_QWORD *)v7 + 120LL);
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v9 = *(__int64 **)(a2 - 8);
      v10 = *v9;
      v11 = v9 + 3;
    }
    else
    {
      v10 = 0;
      v11 = 0;
    }
    v15[2] = v11;
    v16 = 1283;
    v15[0] = "Label: ";
    v15[3] = v10;
    if ( v8 != nullsub_98 )
    {
      v13 = a5;
      v14 = a4;
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v8)(v7, v15, 1);
      a5 = v13;
      a4 = v14;
    }
  }
  return sub_31D4CF0(a1, a2, a3, a4, a5);
}
