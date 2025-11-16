// Function: sub_39820D0
// Address: 0x39820d0
//
void __fastcall sub_39820D0(__int64 *a1, __int64 a2, unsigned __int16 a3)
{
  __int64 v4; // r13
  void (__fastcall *v5)(__int64, __int64, _QWORD); // r14
  unsigned int v6; // eax
  void (*v7)(void); // rax
  __int64 v8; // rdi
  void (*v9)(); // rax
  __int64 v10; // rdi
  void (*v11)(); // rax
  __int64 *v12; // [rsp+0h] [rbp-40h] BYREF
  const char *v13; // [rsp+8h] [rbp-38h]
  __int16 v14; // [rsp+10h] [rbp-30h]

  switch ( a3 )
  {
    case 0u:
    case 1u:
    case 2u:
    case 3u:
    case 4u:
    case 5u:
    case 6u:
    case 7u:
    case 8u:
    case 9u:
    case 0xAu:
    case 0xBu:
    case 0xCu:
    case 0xEu:
    case 0x10u:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x22u:
    case 0x23u:
    case 0x24u:
    case 0x25u:
    case 0x26u:
    case 0x27u:
    case 0x28u:
    case 0x29u:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
      goto LABEL_5;
    case 0xDu:
      if ( *(_BYTE *)(a2 + 416) )
      {
        v8 = *(_QWORD *)(a2 + 256);
        v9 = *(void (**)())(*(_QWORD *)v8 + 104LL);
        v12 = a1;
        v13 = " [signed LEB]";
        v14 = 779;
        if ( v9 != nullsub_580 )
          ((void (__fastcall *)(__int64, __int64 **, __int64))v9)(v8, &v12, 1);
      }
      sub_397C040(a2, *a1, 0);
      return;
    case 0xFu:
    case 0x15u:
    case 0x1Au:
      goto LABEL_3;
    case 0x19u:
    case 0x21u:
      v7 = *(void (**)(void))(**(_QWORD **)(a2 + 256) + 144LL);
      if ( v7 != nullsub_581 )
        v7();
      return;
    default:
      if ( a3 > 0x1F02u )
      {
LABEL_5:
        v4 = *(_QWORD *)(a2 + 256);
        v5 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v4 + 424LL);
        v6 = sub_3982020((unsigned __int64 *)a1, a2, a3);
        v5(v4, *a1, v6);
      }
      else
      {
LABEL_3:
        if ( *(_BYTE *)(a2 + 416) )
        {
          v10 = *(_QWORD *)(a2 + 256);
          v11 = *(void (**)())(*(_QWORD *)v10 + 104LL);
          v13 = " [unsigned LEB]";
          v12 = a1;
          v14 = 779;
          if ( v11 != nullsub_580 )
            ((void (__fastcall *)(__int64, __int64 **, __int64))v11)(v10, &v12, 1);
        }
        sub_397C0C0(a2, *a1, 0);
      }
      return;
  }
}
