// Function: sub_1443560
// Address: 0x1443560
//
__int64 __fastcall sub_1443560(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v6; // r8
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 *v9; // rdx
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v16; // rcx
  __int64 v17; // r8
  int v18; // edx
  int v19; // r10d

  v3 = a1[3];
  v4 = *(unsigned int *)(v3 + 48);
  if ( !(_DWORD)v4 )
    return 0;
  v6 = (unsigned int)(v4 - 1);
  v7 = *(_QWORD *)(v3 + 32);
  v8 = (unsigned int)v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16 * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    v18 = 1;
    while ( v10 != -8 )
    {
      v19 = v18 + 1;
      v8 = (unsigned int)v6 & (v18 + (_DWORD)v8);
      v9 = (__int64 *)(v7 + 16LL * (unsigned int)v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v18 = v19;
    }
    return 0;
  }
LABEL_3:
  if ( v9 == (__int64 *)(v7 + 16 * v4) || !v9[1] )
    return 0;
  v11 = a1[4];
  if ( !v11 )
    return 1;
  v12 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !(unsigned __int8)sub_15CC8F0(v3, v12, a2, v8, v6) )
    return 0;
  if ( (unsigned __int8)sub_15CC8F0(a1[3], v11, a2, v13, v14) )
    return (unsigned int)sub_15CC8F0(a1[3], v12, v11, v16, v17) ^ 1;
  else
    return 1;
}
