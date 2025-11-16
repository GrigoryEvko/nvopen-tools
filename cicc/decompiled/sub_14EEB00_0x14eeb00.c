// Function: sub_14EEB00
// Address: 0x14eeb00
//
__int64 *__fastcall sub_14EEB00(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  const char *v6; // rax
  const char *v8; // [rsp+0h] [rbp-30h] BYREF
  char v9; // [rsp+10h] [rbp-20h]
  char v10; // [rsp+11h] [rbp-1Fh]

  if ( *(_BYTE *)(a4 + 8) != 15 )
  {
    v10 = 1;
    v6 = "Load/Store operand is not a pointer type";
    goto LABEL_5;
  }
  v5 = *(_QWORD *)(a4 + 24);
  if ( a3 && a3 != v5 )
  {
    v10 = 1;
    v6 = "Explicit load/store type does not match pointee type of pointer operand";
LABEL_5:
    v8 = v6;
    v9 = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v8);
    return a1;
  }
  if ( !(unsigned __int8)sub_1643F80(v5) )
  {
    v10 = 1;
    v6 = "Cannot load/store from pointer";
    goto LABEL_5;
  }
  *a1 = 1;
  return a1;
}
