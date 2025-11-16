// Function: sub_15A4510
// Address: 0x15a4510
//
__int64 __fastcall sub_15A4510(__int64 ***a1, __int64 **a2, char a3)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int128 v11; // [rsp-88h] [rbp-88h]
  _QWORD v12[10]; // [rsp-50h] [rbp-50h] BYREF

  result = (__int64)a1;
  if ( *a1 != a2 )
  {
    v12[0] = a1;
    result = sub_1582CC0(0x2Fu, (unsigned __int64)a1, a2);
    if ( !result && !a3 )
    {
      v9 = *a2;
      v12[1] = 47;
      v12[3] = 1;
      v10 = *v9;
      memset(&v12[4], 0, 24);
      v12[2] = v12;
      *((_QWORD *)&v11 + 1) = v12;
      *(_QWORD *)&v11 = 47;
      return sub_15A2780(v10 + 1776, (__int64)a2, v5, v6, v7, v8, v11, 1u, 0);
    }
  }
  return result;
}
