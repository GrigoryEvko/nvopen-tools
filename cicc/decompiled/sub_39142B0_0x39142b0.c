// Function: sub_39142B0
// Address: 0x39142b0
//
unsigned __int64 __fastcall sub_39142B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  int v5; // eax
  int v7; // ecx
  __int64 v8; // rsi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned int *v14; // rbx
  int v16; // eax
  int v17; // r8d

  v4 = 0;
  v5 = *(_DWORD *)(a1 + 104);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = *(_QWORD *)(a1 + 88);
    v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_3:
      v4 = v10[1];
    }
    else
    {
      v16 = 1;
      while ( v11 != -8 )
      {
        v17 = v16 + 1;
        v9 = v7 & (v16 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_3;
        v16 = v17;
      }
      v4 = 0;
    }
  }
  v12 = sub_38D04A0((_QWORD *)a3, a2);
  v13 = (unsigned int)(*(_DWORD *)(a2 + 32) + 1);
  if ( (unsigned int)v13 >= *(_DWORD *)(a3 + 16) )
    return 0;
  v14 = *(unsigned int **)(*(_QWORD *)(a3 + 8) + 8 * v13);
  if ( (*(unsigned __int8 (__fastcall **)(unsigned int *))(*(_QWORD *)v14 + 16LL))(v14) )
    return 0;
  else
    return v14[6] * (((unsigned __int64)v14[6] + v12 + v4 - 1) / v14[6]) - (v12 + v4);
}
