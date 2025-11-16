// Function: sub_2549A80
// Address: 0x2549a80
//
__int64 __fastcall sub_2549A80(__int64 a1, __int64 a2)
{
  char v4; // al
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rdi
  void *v8; // rax
  __int64 v9; // rdx
  const void *v10; // rsi
  unsigned __int8 v11; // [rsp+Eh] [rbp-32h] BYREF
  char v12; // [rsp+Fh] [rbp-31h] BYREF
  _QWORD v13[5]; // [rsp+10h] [rbp-30h] BYREF

  v13[0] = a2;
  v11 = 0;
  v13[1] = a1;
  v13[2] = &v11;
  v12 = 0;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_2595860,
                          (__int64)v13,
                          a1,
                          1u,
                          &v12) )
    return v11 ^ 1u;
  v4 = *(_BYTE *)(a1 + 96);
  v5 = *(unsigned int *)(a1 + 168);
  *(_BYTE *)(a1 + 176) = 1;
  v6 = *(_QWORD *)(a1 + 152);
  *(_BYTE *)(a1 + 136) = v4;
  sub_C7D6A0(v6, 16 * v5, 8);
  v7 = *(unsigned int *)(a1 + 128);
  *(_DWORD *)(a1 + 168) = v7;
  if ( (_DWORD)v7 )
  {
    v8 = (void *)sub_C7D670(16 * v7, 8);
    v9 = *(unsigned int *)(a1 + 168);
    v10 = *(const void **)(a1 + 112);
    *(_QWORD *)(a1 + 152) = v8;
    *(_QWORD *)(a1 + 160) = *(_QWORD *)(a1 + 120);
    memcpy(v8, v10, 16 * v9);
  }
  else
  {
    *(_QWORD *)(a1 + 152) = 0;
    *(_QWORD *)(a1 + 160) = 0;
  }
  return 0;
}
