// Function: sub_28940A0
// Address: 0x28940a0
//
__int64 __fastcall sub_28940A0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 result; // rax

  v3 = *(_QWORD **)(a3 + 24);
  if ( *(_DWORD *)(a3 + 32) > 0x40u )
    v3 = (_QWORD *)*v3;
  v4 = *(_QWORD **)(a2 + 24);
  if ( *(_DWORD *)(a2 + 32) > 0x40u )
    v4 = (_QWORD *)*v4;
  *(_DWORD *)(a1 + 4) = (_DWORD)v3;
  result = (unsigned int)dword_5003CC8;
  *(_DWORD *)a1 = (_DWORD)v4;
  *(_BYTE *)(a1 + 8) = (_DWORD)result == 0;
  return result;
}
