// Function: sub_B0EF30
// Address: 0xb0ef30
//
__int64 __fastcall sub_B0EF30(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, char a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 *v7; // r13
  unsigned int v8; // r12d
  __int64 v9; // r8
  int v10; // ebx
  int v11; // ebx
  int v12; // r13d
  unsigned int i; // r12d
  __int64 v14; // r15
  _BYTE *v15; // rax
  unsigned int v16; // ecx
  _BYTE *v17; // rax
  __int64 v18; // r11
  __int64 v19; // rbx
  __int64 result; // rax
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+30h] [rbp-50h]
  __int64 v29; // [rsp+40h] [rbp-40h] BYREF
  __int64 v30[7]; // [rsp+48h] [rbp-38h] BYREF

  v5 = a3;
  v6 = a2;
  v7 = a1;
  v8 = a4;
  if ( a4 )
  {
LABEL_12:
    v21 = *v7;
    v29 = v6;
    v30[0] = v5;
    v22 = v21 + 760;
    v23 = sub_B97910(16, 2, v8);
    v24 = v23;
    if ( v23 )
      sub_B971C0(v23, (_DWORD)v7, 8, v8, (unsigned int)&v29, 2, 0, 0);
    return sub_B0ED90(v24, v8, v22);
  }
  v9 = *a1;
  v29 = a2;
  v30[0] = a3;
  v10 = *(_DWORD *)(v9 + 784);
  v27 = *(_QWORD *)(v9 + 768);
  if ( !v10 )
    goto LABEL_11;
  v11 = v10 - 1;
  v25 = v9;
  v12 = 1;
  for ( i = v11 & sub_AF7B60(&v29, v30); ; i = v11 & v16 )
  {
    v14 = *(_QWORD *)(v27 + 8LL * i);
    if ( v14 == -4096 )
    {
      v7 = a1;
      v6 = a2;
      v5 = a3;
      v8 = 0;
      goto LABEL_11;
    }
    if ( v14 != -8192 )
    {
      v15 = sub_A17150((_BYTE *)(v14 - 16));
      if ( v29 == *(_QWORD *)v15 )
      {
        v17 = sub_A17150((_BYTE *)(v14 - 16));
        if ( v30[0] == *((_QWORD *)v17 + 1) )
          break;
      }
    }
    v16 = i + v12++;
  }
  v18 = v27 + 8LL * i;
  v19 = v14;
  v7 = a1;
  v6 = a2;
  v5 = a3;
  v8 = 0;
  if ( v18 == *(_QWORD *)(v25 + 768) + 8LL * *(unsigned int *)(v25 + 784) || (result = v19) == 0 )
  {
LABEL_11:
    result = 0;
    if ( !a5 )
      return result;
    goto LABEL_12;
  }
  return result;
}
