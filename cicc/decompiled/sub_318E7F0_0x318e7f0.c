// Function: sub_318E7F0
// Address: 0x318e7f0
//
__int64 *__fastcall sub_318E7F0(__int64 *a1, _QWORD *a2, unsigned int a3)
{
  __int64 v3; // rbx
  __int64 (__fastcall *v4)(__int64); // rax
  __int64 v5; // rax
  int v6; // edx
  char v7; // cl
  __int64 v8; // rax
  __int64 v9; // rdx

  v3 = a3;
  v4 = *(__int64 (__fastcall **)(__int64))(*a2 + 64LL);
  if ( v4 == sub_3184E90 )
  {
    v5 = a2[2];
    v6 = 0;
    if ( (unsigned __int8)(*(_BYTE *)v5 - 22) > 6u )
      v6 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
  }
  else
  {
    v6 = v4((__int64)a2);
    v5 = a2[2];
  }
  v7 = *(_BYTE *)(v5 + 7) & 0x40;
  if ( v6 == (_DWORD)v3 )
  {
    if ( v7 )
      v5 = *(_QWORD *)(v5 - 8) + 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
  }
  else
  {
    if ( v7 )
      v8 = *(_QWORD *)(v5 - 8);
    else
      v8 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
    v5 = 32 * v3 + v8;
  }
  v9 = a2[3];
  *a1 = v5;
  a1[1] = (__int64)a2;
  a1[2] = v9;
  return a1;
}
