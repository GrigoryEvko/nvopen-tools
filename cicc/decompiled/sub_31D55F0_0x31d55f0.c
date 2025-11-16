// Function: sub_31D55F0
// Address: 0x31d55f0
//
const char *__fastcall sub_31D55F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 (*v5)(void); // rax
  __int64 v6; // rdx
  _BYTE *v7; // r8
  __int64 v8; // rax
  __int64 v9; // rdx
  const char *(*v10)(); // rdx
  const char *v11; // r12
  __int64 v13; // rdx
  _BYTE *v14; // [rsp+8h] [rbp-B8h]
  _DWORD v15[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v16; // [rsp+18h] [rbp-A8h]
  _BYTE *v17; // [rsp+20h] [rbp-A0h]
  __int64 v18; // [rsp+28h] [rbp-98h]
  _BYTE v19[144]; // [rsp+30h] [rbp-90h] BYREF

  v4 = 0;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) + 16LL) + 128LL);
  if ( v5 != sub_2DAC790 )
    v4 = v5();
  v6 = *(unsigned __int16 *)(a1 + 68);
  v15[1] = 0;
  v18 = 0x600000000LL;
  v7 = v19;
  v15[0] = v6;
  v8 = v6;
  v9 = *a2;
  v16 = 0;
  v10 = *(const char *(**)())(v9 + 200);
  v17 = v19;
  if ( v10 != sub_C13F30 )
  {
    v11 = (const char *)((__int64 (__fastcall *)(__int64 *, _DWORD *, const char *(*)(), __int64, _BYTE *))v10)(
                          a2,
                          v15,
                          v10,
                          a4,
                          v19);
    if ( v13 )
    {
      v7 = v17;
      goto LABEL_6;
    }
    v8 = *(unsigned __int16 *)(a1 + 68);
    v7 = v17;
  }
  v11 = (const char *)(*(_QWORD *)(v4 + 24) + *(unsigned int *)(*(_QWORD *)(v4 + 16) + 4 * v8));
  if ( v11 )
  {
    v14 = v7;
    strlen(v11);
    v7 = v14;
  }
LABEL_6:
  if ( v7 != v19 )
    _libc_free((unsigned __int64)v7);
  return v11;
}
