// Function: sub_270F340
// Address: 0x270f340
//
__int64 __fastcall sub_270F340(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v6; // edx
  __int64 *v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rbx
  _QWORD *v11; // r12
  __int64 v12; // r12
  char *v13; // r12
  __int64 result; // rax
  int v15; // r8d
  _BYTE v16[16]; // [rsp+0h] [rbp-50h] BYREF
  __int64 (__fastcall *v17)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v3 )
  {
    v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v4 + 16 * v3) )
      {
        v9 = v7[1];
        v10 = *(_QWORD *)(v9 + 16);
        if ( v10 )
        {
          do
          {
            v11 = *(_QWORD **)(v10 + 24);
            if ( *(_BYTE *)v11 == 85 && (unsigned int)sub_B49240(*(_QWORD *)(v10 + 24)) == 259 )
            {
              sub_B43D60(v11);
              v9 = v7[1];
              goto LABEL_10;
            }
            v10 = *(_QWORD *)(v10 + 8);
          }
          while ( v10 );
          v9 = v7[1];
        }
LABEL_10:
        v12 = sub_B57640(v9, (__int64 *)6, v9 + 24, 0);
        sub_B47C00(v12, v7[1], 0, 0);
        sub_BD84D0(v7[1], v12);
        sub_B43D60((_QWORD *)v7[1]);
        *v7 = -8192;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
      }
    }
    else
    {
      v15 = 1;
      while ( v8 != -4096 )
      {
        v6 = (v3 - 1) & (v15 + v6);
        v7 = (__int64 *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        ++v15;
      }
    }
  }
  v13 = *(char **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)(a2 + 16) )
  {
    sub_BD84D0(a2, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    return sub_B43D60((_QWORD *)a2);
  }
  else
  {
    sub_B43D60((_QWORD *)a2);
    v17 = 0;
    sub_F5CAB0(v13, 0, 0, (__int64)v16);
    result = (__int64)v17;
    if ( v17 )
      return v17(v16, v16, 3);
  }
  return result;
}
