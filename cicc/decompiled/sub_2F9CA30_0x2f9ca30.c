// Function: sub_2F9CA30
// Address: 0x2f9ca30
//
__int64 __fastcall sub_2F9CA30(unsigned __int64 a1, unsigned __int16 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int16 v4; // r14
  unsigned __int64 v5; // r12
  __int16 v6; // bx
  __int16 v8; // r13
  unsigned __int16 v9; // [rsp+Ch] [rbp-34h] BYREF
  __int16 v10; // [rsp+Eh] [rbp-32h] BYREF
  unsigned __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v12[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = *(_QWORD *)a3;
  v4 = *(_WORD *)(a3 + 8);
  v11 = a1;
  v9 = a2;
  v12[0] = v3;
  v10 = v4;
  sub_FDCA70(&v11, &v9, v12, (unsigned __int16 *)&v10);
  v5 = v11;
  if ( v11 <= v12[0] )
    return 0;
  v6 = v9;
  if ( !v3 || v12[0] )
    return v11 - v12[0];
  v8 = sub_D788C0(v3, v4);
  if ( (unsigned int)sub_D788E0(v5, v6, 1u, v8 + 64) )
    return v11;
  else
    return -1;
}
