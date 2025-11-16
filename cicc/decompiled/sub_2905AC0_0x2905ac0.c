// Function: sub_2905AC0
// Address: 0x2905ac0
//
__int64 __fastcall sub_2905AC0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  _BYTE *v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // r8
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rdi
  __int64 v16; // r9
  int v17; // edi
  int v18; // r10d
  __int64 v19[3]; // [rsp+8h] [rbp-18h] BYREF

  v7 = *a1;
  v19[0] = *a2;
  v8 = *(_BYTE **)sub_1152A40(v7, v19, a3, a4, a5, a6);
  result = 0;
  if ( *v8 <= 0x15u )
  {
    v10 = a1[1];
    v11 = *(_DWORD *)(v10 + 24);
    v12 = *(_QWORD *)(v10 + 8);
    if ( v11 )
    {
      v13 = v11 - 1;
      v14 = (v11 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v15 = (__int64 *)(v12 + 8LL * v14);
      v16 = *v15;
      if ( *a2 == *v15 )
      {
LABEL_4:
        *v15 = -8192;
        --*(_DWORD *)(v10 + 16);
        ++*(_DWORD *)(v10 + 20);
      }
      else
      {
        v17 = 1;
        while ( v16 != -4096 )
        {
          v18 = v17 + 1;
          v14 = v13 & (v17 + v14);
          v15 = (__int64 *)(v12 + 8LL * v14);
          v16 = *v15;
          if ( *a2 == *v15 )
            goto LABEL_4;
          v17 = v18;
        }
      }
    }
    return 1;
  }
  return result;
}
