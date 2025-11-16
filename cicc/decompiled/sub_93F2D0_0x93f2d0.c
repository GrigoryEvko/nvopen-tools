// Function: sub_93F2D0
// Address: 0x93f2d0
//
__int64 __fastcall sub_93F2D0(int a1)
{
  int v1; // ebx
  int v2; // r9d
  int v3; // r10d
  int v4; // r8d
  int v5; // ecx
  int v6; // esi
  const char *v7; // rdi
  int v8; // eax
  int v9; // edx
  int v10; // eax
  __int64 v11; // r12
  int v13; // [rsp+0h] [rbp-1C0h]
  int v14; // [rsp+10h] [rbp-1B0h]
  int v15; // [rsp+18h] [rbp-1A8h]
  int v16; // [rsp+18h] [rbp-1A8h]
  _BOOL4 v17; // [rsp+2Ch] [rbp-194h]
  __int128 v18; // [rsp+50h] [rbp-170h]
  __int128 v19; // [rsp+60h] [rbp-160h]
  _BYTE *v20; // [rsp+70h] [rbp-150h] BYREF
  __int64 v21; // [rsp+78h] [rbp-148h]
  __int64 v22; // [rsp+80h] [rbp-140h]
  _BYTE v23[312]; // [rsp+88h] [rbp-138h] BYREF

  v1 = 1;
  if ( !dword_4D046B4 )
  {
    v1 = 0;
    if ( unk_4D0465C )
      v1 = dword_4D04658 == 0 ? 3 : 0;
  }
  v21 = 0;
  v20 = v23;
  v22 = 256;
  sub_C82800(&v20);
  v3 = a1 + 16;
  v4 = v21;
  v5 = (int)v20;
  BYTE8(v19) = 0;
  v17 = unk_4D04660 != 0;
  v6 = (int)qword_4D046E0;
  v7 = qword_4D046E0;
  if ( qword_4D046E0 || (v6 = (int)qword_4F076F0, (v7 = qword_4F076F0) != 0) )
  {
    v13 = (int)v20;
    v14 = v21;
    v15 = v3;
    v8 = strlen(v7);
    v3 = v15;
    v4 = v14;
    v5 = v13;
    v9 = v8;
  }
  else
  {
    v9 = 0;
  }
  v16 = v3;
  v10 = sub_ADC750(v3, v6, v9, v5, v4, v2, v18, v19);
  v11 = sub_ADDEF0(
          v16,
          4,
          v10,
          (unsigned int)"lgenfe: EDG 6.6",
          15,
          v17,
          (__int64)byte_3F871B3,
          0,
          0,
          (__int64)byte_3F871B3,
          0,
          v1,
          0,
          1,
          0,
          0,
          0,
          0,
          0,
          0,
          0);
  if ( v20 != v23 )
    _libc_free(v20, 4);
  return v11;
}
