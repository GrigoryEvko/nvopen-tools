// Function: sub_39F8020
// Address: 0x39f8020
//
__int64 __fastcall sub_39F8020(__int64 a1, __int64 a2)
{
  char *v2; // rcx
  __int64 i; // rax
  int v4; // edx
  __int64 v5; // rdx
  _QWORD v7[30]; // [rsp+0h] [rbp-288h] BYREF
  char v8[296]; // [rsp+F0h] [rbp-198h] BYREF
  __int64 v9; // [rsp+218h] [rbp-70h]
  __int64 v10; // [rsp+220h] [rbp-68h]
  int v11; // [rsp+230h] [rbp-58h]
  __int64 v12; // [rsp+258h] [rbp-30h]
  __int64 v13; // [rsp+268h] [rbp-20h]

  memset(v7, 0, sizeof(v7));
  v7[19] = a1 + 1;
  v7[24] = 0x4000000000000000LL;
  if ( (unsigned int)sub_39F7420(v7, v8) || v11 == 2 )
    return 0;
  v2 = v8;
  for ( i = 0; i != 18; ++i )
  {
    while ( 1 )
    {
      v4 = *((_DWORD *)v2 + 2);
      *(_BYTE *)(a2 + i + 180) = v4;
      if ( (_BYTE)v4 == 1 || (_BYTE)v4 == 2 )
        break;
      *(_QWORD *)(a2 + 8 * i++ + 32) = 0;
      v2 += 16;
      if ( i == 18 )
        goto LABEL_8;
    }
    v5 = *(_QWORD *)v2;
    v2 += 16;
    *(_QWORD *)(a2 + 8 * i + 32) = v5;
  }
LABEL_8:
  *(_QWORD *)(a2 + 16) = v9;
  *(_WORD *)(a2 + 176) = v10;
  *(_WORD *)(a2 + 178) = v12;
  *(_QWORD *)(a2 + 24) = v7[26];
  *(_QWORD *)(a2 + 8) = v13;
  return a2;
}
