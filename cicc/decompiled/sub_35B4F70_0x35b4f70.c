// Function: sub_35B4F70
// Address: 0x35b4f70
//
__int64 __fastcall sub_35B4F70(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rsi
  int v3; // r12d
  int v4; // ebx
  __int64 v5; // r14
  unsigned int v6; // edx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r15
  unsigned __int64 v10; // rax
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // r9
  unsigned __int64 v16; // r10
  __int64 *v17; // rdx
  __int64 *v18; // rsi
  unsigned __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  _QWORD *v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  int v23[2]; // [rsp+28h] [rbp-38h] BYREF

  result = sub_CA08F0(
             (__int64 *)v23,
             "seed",
             4u,
             (__int64)"Seed Live Regs",
             14,
             byte_4F826E9[0],
             "regalloc",
             8u,
             "Register Allocation",
             19);
  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(_DWORD *)(v2 + 64);
  if ( v3 )
  {
    v4 = 0;
    while ( 1 )
    {
      v6 = v4 & 0x7FFFFFFF;
      v7 = v4 & 0x7FFFFFFF;
      result = *(_QWORD *)(*(_QWORD *)(v2 + 56) + 16 * v7 + 8);
      if ( !result )
        goto LABEL_5;
      if ( (*(_BYTE *)(result + 4) & 8) == 0 )
        break;
      while ( 1 )
      {
        result = *(_QWORD *)(result + 32);
        if ( !result )
          break;
        if ( (*(_BYTE *)(result + 4) & 8) == 0 )
          goto LABEL_9;
      }
      if ( v3 == ++v4 )
        goto LABEL_16;
LABEL_6:
      v2 = *(_QWORD *)(a1 + 16);
    }
LABEL_9:
    v8 = *(_QWORD *)(a1 + 32);
    v9 = 8 * v7;
    v10 = *(unsigned int *)(v8 + 160);
    if ( (unsigned int)v10 > v6 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(v8 + 152) + 8 * v7);
      if ( v5 )
      {
LABEL_4:
        result = sub_35B4EE0(a1, v5);
LABEL_5:
        if ( v3 == ++v4 )
          goto LABEL_16;
        goto LABEL_6;
      }
    }
    v11 = v6 + 1;
    if ( (unsigned int)v10 < v11 && v11 != v10 )
    {
      if ( v11 >= v10 )
      {
        v15 = *(_QWORD *)(v8 + 168);
        v16 = v11 - v10;
        if ( v11 > (unsigned __int64)*(unsigned int *)(v8 + 164) )
        {
          v19 = v11 - v10;
          v20 = *(_QWORD *)(v8 + 168);
          v22 = *(_QWORD *)(a1 + 32);
          sub_C8D5F0(v8 + 152, (const void *)(v8 + 168), v11, 8u, v8, v15);
          v8 = v22;
          v16 = v19;
          v15 = v20;
          v10 = *(unsigned int *)(v22 + 160);
        }
        v12 = *(_QWORD *)(v8 + 152);
        v17 = (__int64 *)(v12 + 8 * v10);
        v18 = &v17[v16];
        if ( v17 != v18 )
        {
          do
            *v17++ = v15;
          while ( v18 != v17 );
          LODWORD(v10) = *(_DWORD *)(v8 + 160);
          v12 = *(_QWORD *)(v8 + 152);
        }
        *(_DWORD *)(v8 + 160) = v16 + v10;
        goto LABEL_12;
      }
      *(_DWORD *)(v8 + 160) = v11;
    }
    v12 = *(_QWORD *)(v8 + 152);
LABEL_12:
    v21 = (_QWORD *)v8;
    v13 = (__int64 *)(v12 + v9);
    v14 = sub_2E10F30(v4 | 0x80000000);
    *v13 = v14;
    v5 = v14;
    sub_2E11E80(v21, v14);
    goto LABEL_4;
  }
LABEL_16:
  if ( *(_QWORD *)v23 )
    return sub_C9E2A0(*(__int64 *)v23);
  return result;
}
