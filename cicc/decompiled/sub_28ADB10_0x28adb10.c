// Function: sub_28ADB10
// Address: 0x28adb10
//
__int64 __fastcall sub_28ADB10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v7; // rdi
  unsigned int v8; // r15d
  unsigned __int8 *v9; // r15
  unsigned __int8 *v10; // rax
  __int64 v11; // rax

  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( **(_BYTE **)(a2 + 32 * (2 - v3)) != 17 )
    return 0;
  v7 = *(_QWORD *)(a2 + 32 * (3 - v3));
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 <= 0x40 )
  {
    if ( *(_QWORD *)(v7 + 24) )
      return 0;
  }
  else if ( v8 != (unsigned int)sub_C444A0(v7 + 24) )
  {
    return 0;
  }
  v9 = *(unsigned __int8 **)(a2 + 32 * (1 - v3));
  v10 = sub_BD3990(*(unsigned __int8 **)(a2 - 32 * v3), a2);
  v11 = sub_28AD0D0(a1, a2, (__int64)v10, v9);
  if ( !v11 )
    return 0;
  *(_QWORD *)a3 = v11 + 24;
  *(_WORD *)(a3 + 8) = 0;
  return 1;
}
