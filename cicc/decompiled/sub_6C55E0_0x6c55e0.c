// Function: sub_6C55E0
// Address: 0x6c55e0
//
__int64 __fastcall sub_6C55E0(__int64 a1, unsigned int a2, __int64 a3, int a4, __m128i *a5, __int64 *a6)
{
  int v7; // r14d
  __int16 v8; // r13
  bool v9; // bl
  __int64 result; // rax
  unsigned int v11; // [rsp+Ch] [rbp-2Ch] BYREF
  _QWORD v12[5]; // [rsp+10h] [rbp-28h] BYREF

  v7 = dword_4F077C8;
  v8 = unk_4F077CC;
  v9 = a2 == 0 && a1 == 0;
  sub_6C0910(0, 0, 1u, v12, 0, 1, 0, a4, a1, a2, a3, 0, a5, &v11, 0);
  if ( v9 )
  {
    v7 = dword_4F061D8;
    v8 = unk_4F061DC;
  }
  result = v11;
  if ( v11 )
  {
    *a6 = 0;
  }
  else
  {
    result = sub_6F5430(0, v12[0], 0, 0, 0, 0, 0, 0, 0, 0, 0);
    *a6 = result;
  }
  if ( v9 )
  {
    result = (__int64)&dword_4F061D8;
    dword_4F061D8 = v7;
    unk_4F061DC = v8;
  }
  return result;
}
