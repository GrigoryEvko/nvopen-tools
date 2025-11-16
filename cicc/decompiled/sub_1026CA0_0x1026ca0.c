// Function: sub_1026CA0
// Address: 0x1026ca0
//
__int64 __fastcall sub_1026CA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // r9
  unsigned int v8; // edx
  __int64 *v9; // rcx
  __int64 v10; // rdi
  int v11; // ecx
  int v12; // r11d
  _QWORD v13[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( (_BYTE)qword_4F8EE08 )
    return 0;
  v5 = *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v5 )
    return 0;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v6 + 40LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    v11 = 1;
    while ( v10 != -4096 )
    {
      v12 = v11 + 1;
      v8 = (v5 - 1) & (v11 + v8);
      v9 = (__int64 *)(v6 + 40LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_5;
      v11 = v12;
    }
    return 0;
  }
LABEL_5:
  if ( v9 == (__int64 *)(v6 + 40 * v5) )
    return 0;
  result = 1;
  if ( v9[3] )
  {
    v13[1] = v3;
    v13[0] = a3;
    return ((__int64 (__fastcall *)(__int64 *, _QWORD *))v9[4])(v9 + 1, v13);
  }
  return result;
}
