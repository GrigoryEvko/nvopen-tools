// Function: sub_37378D0
// Address: 0x37378d0
//
__int64 __fastcall sub_37378D0(__int64 *a1, __int64 a2, __int16 a3, unsigned int a4)
{
  __int64 v4; // r15
  unsigned __int64 **v5; // r14
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 result; // rax
  unsigned __int16 v14; // ax
  __int16 v15; // dx
  unsigned int v16; // r15d
  __int16 v17; // [rsp+4h] [rbp-4Ch]
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int16 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]

  v4 = a4;
  v5 = (unsigned __int64 **)(a2 + 8);
  v9 = a1[23];
  v10 = a1[26];
  v11 = *(_QWORD *)(v9 + 200);
  if ( (unsigned int)(*(_DWORD *)(v11 + 544) - 42) > 1 )
  {
    v14 = sub_3220AA0(v10);
    v15 = 34;
    if ( v14 <= 4u )
      v15 = sub_3222A40(a1[26]);
    if ( !a3
      || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
      || (v17 = v15, v19 = sub_3220AA0(a1[26]), result = sub_E06A90(a3), v15 = v17, v19 >= (unsigned int)result) )
    {
      HIWORD(v20) = v15;
      WORD2(v20) = a3;
      v21 = v4;
      LODWORD(v20) = 10;
      return sub_3248F80(v5, a1 + 11, &v20);
    }
  }
  else
  {
    v12 = *(_QWORD *)(*(_QWORD *)(v10 + 1288) + 32LL * a4 + 24);
    if ( !a3
      || (*(_BYTE *)(v11 + 904) & 0x40) == 0
      || (v18 = *(_QWORD *)(*(_QWORD *)(v10 + 1288) + 32 * v4 + 24),
          v16 = (unsigned __int16)sub_3220AA0(v10),
          result = sub_E06A90(a3),
          v12 = v18,
          v16 >= (unsigned int)result) )
    {
      LODWORD(v20) = 1;
      WORD2(v20) = a3;
      HIWORD(v20) = 6;
      v21 = v12;
      return sub_3248F80(v5, a1 + 11, &v20);
    }
  }
  return result;
}
