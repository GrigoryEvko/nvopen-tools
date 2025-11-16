// Function: sub_2D42CA0
// Address: 0x2d42ca0
//
void __fastcall sub_2D42CA0(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rax
  _BYTE *v5; // rdi
  __int64 *v6; // r14
  __int64 v7; // rbx
  _BYTE *v8; // r15
  unsigned int v9; // ebx
  __int64 v10; // r12
  _BYTE *v11; // [rsp+8h] [rbp-C8h]
  _BYTE *v12; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v13; // [rsp+18h] [rbp-B8h]
  _BYTE v14[176]; // [rsp+20h] [rbp-B0h] BYREF

  v3 = *(_QWORD *)(a2 + 48) == 0;
  v12 = v14;
  v13 = 0x800000000LL;
  if ( !v3 || (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
    sub_B9AA80(a2, (__int64)&v12);
  v4 = sub_BD5C60(a1);
  v5 = v12;
  v6 = (__int64 *)v4;
  v7 = 16LL * (unsigned int)v13;
  v11 = &v12[v7];
  if ( &v12[v7] != v12 )
  {
    v8 = v12;
    while ( 2 )
    {
      v9 = *(_DWORD *)v8;
      v10 = *((_QWORD *)v8 + 1);
      switch ( *(_DWORD *)v8 )
      {
        case 0:
        case 1:
        case 5:
        case 7:
        case 8:
        case 0x19:
        case 0x28:
        case 0x29:
          goto LABEL_13;
        default:
          if ( (unsigned int)sub_B6ED60(v6, "amdgpu.no.remote.memory", 0x17u) == v9
            || (unsigned int)sub_B6ED60(v6, "amdgpu.no.fine.grained.memory", 0x1Du) == v9 )
          {
LABEL_13:
            sub_B99FD0(a1, v9, v10);
          }
          v8 += 16;
          if ( v11 != v8 )
            continue;
          v5 = v12;
          break;
      }
      break;
    }
  }
  if ( v5 != v14 )
    _libc_free((unsigned __int64)v5);
}
