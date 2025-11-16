// Function: sub_321F6F0
// Address: 0x321f6f0
//
__int64 __fastcall sub_321F6F0(__int64 a1)
{
  __int64 *v1; // r13
  __int64 result; // rax
  __int64 *v3; // r12
  int v5; // edi
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // r8
  __int64 v9; // rdi
  __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  int v14; // eax
  int v15; // eax
  int v16; // r9d

  v1 = *(__int64 **)(a1 + 760);
  result = *(unsigned int *)(a1 + 768);
  v3 = &v1[result];
  if ( v1 != v3 )
  {
    while ( 1 )
    {
      v11 = sub_32150B0(*(_QWORD *)(*v1 + 24));
      v12 = *(_QWORD *)(a1 + 680);
      v13 = v11;
      v14 = *(_DWORD *)(a1 + 696);
      if ( !v14 )
        goto LABEL_7;
      v5 = v14 - 1;
      v6 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v7 = (__int64 *)(v12 + 16LL * v6);
      v8 = *v7;
      if ( v13 != *v7 )
        break;
LABEL_4:
      v9 = v7[1];
LABEL_5:
      v10 = *v1++;
      result = sub_3739390(v9, v10);
      if ( v3 == v1 )
        return result;
    }
    v15 = 1;
    while ( v8 != -4096 )
    {
      v16 = v15 + 1;
      v6 = v5 & (v15 + v6);
      v7 = (__int64 *)(v12 + 16LL * v6);
      v8 = *v7;
      if ( v13 == *v7 )
        goto LABEL_4;
      v15 = v16;
    }
LABEL_7:
    v9 = 0;
    goto LABEL_5;
  }
  return result;
}
