// Function: sub_35EE2E0
// Address: 0x35ee2e0
//
__int64 __fastcall sub_35EE2E0(__int64 a1, int a2)
{
  __int64 v2; // rbp
  __int64 result; // rax
  _QWORD *v4; // [rsp-78h] [rbp-78h] BYREF
  __int16 v5; // [rsp-58h] [rbp-58h]
  _QWORD v6[4]; // [rsp-48h] [rbp-48h] BYREF
  char v7; // [rsp-28h] [rbp-28h]
  void *v8; // [rsp-20h] [rbp-20h] BYREF
  int v9; // [rsp-18h] [rbp-18h]
  _QWORD v10[2]; // [rsp-10h] [rbp-10h] BYREF

  result = a1;
  switch ( a2 )
  {
    case 0:
      *(_BYTE *)(a1 + 22) = 99;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1701733735;
      *(_WORD *)(a1 + 20) = 26994;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      return result;
    case 1:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "global");
      *(_QWORD *)(a1 + 8) = 6;
      return result;
    case 2:
      goto LABEL_9;
    case 3:
      strcpy((char *)(a1 + 16), "shared");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      return result;
    case 4:
      *(_DWORD *)(a1 + 16) = 1936617315;
      *(_QWORD *)a1 = a1 + 16;
      *(_BYTE *)(a1 + 20) = 116;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return result;
    case 5:
      *(_DWORD *)(a1 + 16) = 1633906540;
      *(_QWORD *)a1 = a1 + 16;
      *(_BYTE *)(a1 + 20) = 108;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return result;
    default:
      if ( a2 != 101 )
      {
LABEL_9:
        v10[1] = v2;
        v6[0] = "Unknown NVPTX::AddressSpace \"{}\".";
        v6[2] = v10;
        v9 = a2;
        v6[1] = 33;
        v8 = &unk_49E65E8;
        v6[3] = 1;
        v7 = 1;
        v10[0] = &v8;
        v5 = 263;
        v4 = v6;
        sub_C64D30((__int64)&v4, 1u);
      }
      *(_DWORD *)(a1 + 16) = 1634886000;
      *(_QWORD *)a1 = a1 + 16;
      *(_BYTE *)(a1 + 20) = 109;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return result;
  }
}
