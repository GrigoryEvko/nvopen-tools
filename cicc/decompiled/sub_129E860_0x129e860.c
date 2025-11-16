// Function: sub_129E860
// Address: 0x129e860
//
__int64 __fastcall sub_129E860(int a1)
{
  int v1; // ebx
  int v2; // r8d
  int v3; // ecx
  int v4; // esi
  const char *v5; // rdi
  int v6; // eax
  int v7; // edx
  int v8; // eax
  __int64 v9; // r12
  int v11; // [rsp+8h] [rbp-1A8h]
  int v12; // [rsp+18h] [rbp-198h]
  _BOOL4 v13; // [rsp+28h] [rbp-188h]
  int v14; // [rsp+2Ch] [rbp-184h]
  _BYTE v15[32]; // [rsp+30h] [rbp-180h] BYREF
  _BYTE v16[32]; // [rsp+50h] [rbp-160h] BYREF
  _BYTE *v17; // [rsp+70h] [rbp-140h] BYREF
  __int64 v18; // [rsp+78h] [rbp-138h]
  _BYTE v19[304]; // [rsp+80h] [rbp-130h] BYREF

  v14 = 1;
  if ( !dword_4D046B4 )
  {
    v14 = 0;
    if ( unk_4D0465C )
      v14 = dword_4D04658 == 0 ? 3 : 0;
  }
  v1 = a1 + 16;
  v17 = v19;
  v18 = 0x10000000000LL;
  sub_16C56A0(&v17);
  v15[16] = 0;
  v16[24] = 0;
  v2 = v18;
  v3 = (int)v17;
  v13 = unk_4D04660 != 0;
  v4 = (int)qword_4D046E0;
  v5 = qword_4D046E0;
  if ( qword_4D046E0 || (v4 = (int)qword_4F076F0, (v5 = qword_4F076F0) != 0) )
  {
    v11 = (int)v17;
    v12 = v18;
    v6 = strlen(v5);
    v2 = v12;
    v3 = v11;
    v7 = v6;
  }
  else
  {
    v7 = 0;
  }
  v8 = sub_15A56E0(v1, v4, v7, v3, v2, (unsigned int)v16, (__int64)v15);
  v9 = sub_15A6C20(
         v1,
         4,
         v8,
         (unsigned int)"lgenfe: EDG 6.6",
         15,
         v13,
         (__int64)byte_3F871B3,
         0,
         0,
         (__int64)byte_3F871B3,
         0,
         v14,
         0,
         1,
         0,
         0);
  if ( v17 != v19 )
    _libc_free(v17, 4);
  return v9;
}
