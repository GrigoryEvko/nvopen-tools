// Function: sub_E9F3F0
// Address: 0xe9f3f0
//
__int64 __fastcall sub_E9F3F0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 (*v8)(); // rdx
  __int64 v9; // r13
  __int64 (*v10)(); // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD v16[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v17; // [rsp+20h] [rbp-30h]

  result = sub_E99590(a1, a2);
  if ( result )
  {
    v6 = result;
    if ( *(_BYTE *)(a1 + 278) )
    {
      v7 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 278) = 0;
      v8 = sub_E97650;
      v9 = 1;
      v10 = *(__int64 (**)())(v7 + 88);
      if ( v10 != sub_E97650 )
        v9 = ((__int64 (__fastcall *)(__int64))v10)(a1);
      result = sub_E9F050(v6 + 112, (__int64 *)(a1 + 280), (__int64)v8, v3, v4, v5);
      *(_QWORD *)(result + 32) = v9;
      *(_QWORD *)(a1 + 280) = 0;
    }
    else
    {
      v11 = *(_QWORD *)(result + 32);
      v12 = *(_QWORD *)(a1 + 8);
      if ( (*(_BYTE *)(v11 + 8) & 1) != 0 )
      {
        v13 = *(__int64 **)(v11 - 8);
        v14 = *v13;
        v15 = v13 + 3;
      }
      else
      {
        v14 = 0;
        v15 = 0;
      }
      v16[3] = v14;
      v17 = 1283;
      v16[0] = "Stray .seh_endepilogue in ";
      v16[2] = v15;
      return sub_E66880(v12, a2, (__int64)v16);
    }
  }
  return result;
}
