// Function: sub_1B9A880
// Address: 0x1b9a880
//
unsigned __int64 __fastcall sub_1B9A880(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int *a5,
        unsigned __int64 *a6)
{
  __int64 v6; // rbp
  unsigned __int64 result; // rax
  unsigned int *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned int v10[2]; // [rsp-10h] [rbp-10h] BYREF
  __int64 v11; // [rsp-8h] [rbp-8h]

  if ( *(_BYTE *)(a3 + 16) != 60 )
  {
    result = *(unsigned int *)(a2 + 56);
    if ( (_DWORD)result )
    {
      v8 = (unsigned int *)(a1 + 280);
      v9 = **(_QWORD **)(a2 + 48);
      if ( (_DWORD)a6 == -1 )
      {
        return sub_1B99BD0(v8, v9, (unsigned int)a5, a4, a5, -1);
      }
      else
      {
        v11 = v6;
        v10[0] = (unsigned int)a5;
        v10[1] = (unsigned int)a6;
        return sub_1B9A1B0(v8, v9, v10, a4, (int)a5, a6);
      }
    }
  }
  return result;
}
