// Function: sub_3729860
// Address: 0x3729860
//
__int64 __fastcall sub_3729860(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r13
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // rax
  __int64 v10; // rax

  result = sub_37F8FF0(a2);
  if ( (_BYTE)result )
  {
    v7 = sub_32530C0((char *)a1, (__int64)a2, v3, v4, v5, v6);
    v8 = (unsigned __int8 *)sub_B2E500(*a2);
    v9 = sub_BD3990(v8, (__int64)a2);
    v10 = sub_23CF390(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL), (__int64)v9);
    return sub_3729480(a1, v7, v10);
  }
  return result;
}
