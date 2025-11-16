// Function: sub_778F10
// Address: 0x778f10
//
__int64 __fastcall sub_778F10(__int64 a1, unsigned __int64 a2, FILE *a3, _QWORD *a4, __int64 a5, void *a6, __int64 a7)
{
  FILE *v7; // r10
  unsigned int v10; // ebx
  char v11; // al
  __int64 result; // rax
  unsigned int v13; // eax
  __int64 v14; // [rsp+0h] [rbp-50h]
  unsigned int v16[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v7 = a3;
  v10 = 16;
  v11 = *(_BYTE *)(a2 + 140);
  v16[0] = 1;
  if ( (unsigned __int8)(v11 - 2) <= 1u
    || (v14 = a5, v13 = sub_7764B0(a1, a2, v16), v7 = a3, a5 = v14, v10 = v13, (result = v16[0]) != 0) )
  {
    result = sub_778A80(a1, a2, v7, a4, a5, (__int64)a6, a7);
    if ( (_DWORD)result )
    {
      memcpy(a6, a4, v10);
      return v16[0];
    }
  }
  return result;
}
