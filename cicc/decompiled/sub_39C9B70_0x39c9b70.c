// Function: sub_39C9B70
// Address: 0x39c9b70
//
unsigned __int64 __fastcall sub_39C9B70(__int64 a1, unsigned __int64 **a2, __int16 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned __int64 v7; // rcx
  unsigned __int64 *v8; // rdx

  result = sub_145CDC0(0x18u, (__int64 *)(a1 + 88));
  if ( result )
  {
    *(_DWORD *)(result + 8) = 3;
    v7 = result;
    *(_WORD *)(result + 14) = a3;
    *(_QWORD *)result = result | 4;
    *(_WORD *)(result + 12) = 0;
    *(_QWORD *)(result + 16) = a4;
  }
  else
  {
    v7 = 0;
  }
  v8 = *a2;
  if ( *a2 )
  {
    *(_QWORD *)result = *v8;
    result = v7 & 0xFFFFFFFFFFFFFFFBLL;
    *v8 = v7 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *a2 = (unsigned __int64 *)v7;
  return result;
}
