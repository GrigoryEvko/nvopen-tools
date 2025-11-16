// Function: sub_899160
// Address: 0x899160
//
__int64 __fastcall sub_899160(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 result; // rax
  __int64 v17; // [rsp+8h] [rbp-E8h] BYREF
  __m128i v18[4]; // [rsp+10h] [rbp-E0h] BYREF
  char v19[8]; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v20; // [rsp+58h] [rbp-98h] BYREF
  char v21; // [rsp+91h] [rbp-5Fh]

  v8 = (_QWORD *)*a1;
  sub_7B8B50((unsigned __int64)a1, a2, a3, a4, a5, a6);
  sub_87E3B0((__int64)v19);
  sub_627530((__int64)v8, 0x100080u, &v17, v19, 0, (__int64)a2, 0, 0, 0, 0, 0, 1, 0, a3);
  v9 = v17;
  if ( *(_BYTE *)(v17 + 140) == 7 )
    *(_QWORD *)(v17 + 160) = v8[34];
  v10 = (_QWORD *)v8[44];
  v8[36] = v9;
  if ( v10 )
  {
    sub_869FD0(v10, dword_4F04C64);
    v8[44] = 0;
  }
  v11 = sub_893A40((__int64)a1, v18, (__int64)v19);
  *v8 = v11;
  sub_8911B0((__int64)a1, (__int64)v11, v12, v13, v14, v15);
  result = dword_4F04C64;
  if ( dword_4F04C64 == -1
    || (result = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(result + 7) & 1) == 0)
    || dword_4F04C44 == -1 && (*(_BYTE *)(result + 6) & 2) == 0 )
  {
    if ( (v21 & 8) == 0 )
      return (__int64)sub_87E280(&v20);
  }
  return result;
}
