// Function: sub_3202550
// Address: 0x3202550
//
void __fastcall sub_3202550(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int16 v4; // ax
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // r12d
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  int v12; // [rsp+8h] [rbp-28h]

  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v3 + 782) && *(_QWORD *)(sub_31DA6B0(v3) + 392) )
  {
    switch ( *(_DWORD *)(a2 + 264) )
    {
      case 3:
        v4 = 246;
        break;
      case 0x11:
        v4 = 16;
        break;
      case 0x24:
        v4 = 244;
        break;
      case 0x26:
        v4 = 7;
        break;
      case 0x27:
        v4 = 208;
        break;
      default:
        sub_C64ED0("target architecture doesn't map to a CodeView CPUType", 1u);
    }
    *(_WORD *)(a1 + 786) = v4;
    v12 = 0;
    v11 = sub_BA8DC0(a2, (__int64)"llvm.dbg.cu", 11);
    sub_BA95A0((__int64)&v11);
    v5 = sub_BA9580((__int64)&v11);
    v6 = 3;
    v7 = (unsigned int)(*(_DWORD *)(v5 + 16) - 1);
    if ( (unsigned int)v7 <= 0x22 )
      v6 = byte_44D4F60[v7];
    *(_BYTE *)(a1 + 800) = v6;
    sub_3201700(a1);
    v8 = sub_BA91D0(a2, "CodeViewGHash", 0xDu);
    if ( v8 )
    {
      v9 = *(_QWORD *)(v8 + 136);
      LOBYTE(v8) = 0;
      if ( v9 )
      {
        v10 = *(_DWORD *)(v9 + 32);
        if ( v10 <= 0x40 )
          LOBYTE(v8) = *(_QWORD *)(v9 + 24) != 0;
        else
          LOBYTE(v8) = v10 != (unsigned int)sub_C444A0(v9 + 24);
      }
    }
    *(_BYTE *)(a1 + 784) = v8;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
  }
}
