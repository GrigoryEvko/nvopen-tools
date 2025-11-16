// Function: sub_E99880
// Address: 0xe99880
//
__int64 __fastcall sub_E99880(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  _QWORD *v3; // rdx
  __int64 v4; // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 *v9; // rdx
  _QWORD v10[4]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v11; // [rsp+20h] [rbp-20h]

  result = sub_E99590(a1, a2);
  if ( result )
  {
    v3 = *(_QWORD **)(result + 40);
    if ( v3 )
    {
      v4 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 278) = 1;
      v5 = *(__int64 (**)())(v4 + 88);
      result = 1;
      if ( v5 != sub_E97650 )
        result = ((__int64 (__fastcall *)(__int64))v5)(a1);
      *(_QWORD *)(a1 + 280) = result;
    }
    else
    {
      v6 = *(_QWORD *)(result + 32);
      v7 = *(_QWORD *)(a1 + 8);
      v8 = 0;
      if ( (*(_BYTE *)(v6 + 8) & 1) != 0 )
      {
        v9 = *(__int64 **)(v6 - 8);
        v8 = *v9;
        v3 = v9 + 3;
      }
      v10[2] = v3;
      v11 = 1283;
      v10[0] = "starting epilogue (.seh_startepilogue) before prologue has ended (.seh_endprologue) in ";
      v10[3] = v8;
      return sub_E66880(v7, a2, (__int64)v10);
    }
  }
  return result;
}
