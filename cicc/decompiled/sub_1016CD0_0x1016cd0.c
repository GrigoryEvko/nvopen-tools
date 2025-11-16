// Function: sub_1016CD0
// Address: 0x1016cd0
//
bool __fastcall sub_1016CD0(unsigned __int64 a1, _BYTE *a2, _BYTE *a3, __int64 *a4, unsigned int a5)
{
  _BYTE *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8

  v5 = (_BYTE *)sub_1012FB0(a1, a2, a3, a4, a5);
  return v5 && *v5 <= 0x15u && sub_AD7930(v5, (__int64)a2, v6, v7, v8);
}
