// Function: sub_35EE190
// Address: 0x35ee190
//
__int64 __fastcall sub_35EE190(__int64 a1, int a2)
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
      strcpy((char *)(a1 + 16), "Thread");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      break;
    case 1:
      *(_DWORD *)(a1 + 16) = 1668246594;
      *(_QWORD *)a1 = a1 + 16;
      *(_BYTE *)(a1 + 20) = 107;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      break;
    case 2:
      *(_BYTE *)(a1 + 22) = 114;
      *(_QWORD *)a1 = a1 + 16;
      *(_DWORD *)(a1 + 16) = 1937075267;
      *(_WORD *)(a1 + 20) = 25972;
      *(_QWORD *)(a1 + 8) = 7;
      *(_BYTE *)(a1 + 23) = 0;
      break;
    case 3:
      strcpy((char *)(a1 + 16), "Device");
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 6;
      break;
    case 4:
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "System");
      *(_QWORD *)(a1 + 8) = 6;
      break;
    default:
      v10[1] = v2;
      v6[0] = "Unknown NVPTX::Scope \"{}\".";
      v6[2] = v10;
      v9 = a2;
      v6[1] = 26;
      v8 = &unk_49E65E8;
      v6[3] = 1;
      v7 = 1;
      v10[0] = &v8;
      v5 = 263;
      v4 = v6;
      sub_C64D30((__int64)&v4, 1u);
  }
  return result;
}
