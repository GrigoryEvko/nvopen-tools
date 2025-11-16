// Function: sub_943430
// Address: 0x943430
//
const char *__fastcall sub_943430(_QWORD *a1, __int64 a2, __int64 a3, const char *a4)
{
  const char *v5; // r12
  const char *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  int v10; // r9d
  const char *v11; // r12
  __int64 v12; // r9
  __int64 v13; // r10
  int v14; // eax
  int v15; // edx
  int v16; // r15d
  int v17; // r14d
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h]
  int v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  int v24; // [rsp+18h] [rbp-48h]
  char v25; // [rsp+1Fh] [rbp-41h]
  char v26[52]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = a4;
  if ( !a4 )
  {
    v5 = *(const char **)(a3 + 8);
    if ( !v5 )
      v5 = byte_3F871B3;
  }
  sub_93ED80(*(_DWORD *)(a3 + 64), v26);
  result = (const char *)sub_93EC00((__int64)v5, a3);
  v11 = result;
  if ( *result )
  {
    v12 = sub_9405D0((__int64)a1, *(_DWORD *)(a3 + 64), v7, v8, v9, v10);
    v25 = *(_BYTE *)(a3 + 136) == 2;
    if ( *(_BYTE *)(a3 + 136) != 2 || (v20 = a1[64], a1[60] == v20) )
    {
      v13 = a1[1];
    }
    else
    {
      if ( v20 == a1[65] )
        v20 = *(_QWORD *)(a1[67] - 8LL) + 512LL;
      v13 = *(_QWORD *)(v20 - 8);
    }
    v21 = v12;
    v22 = v13;
    v23 = sub_941B90((__int64)a1, *(_QWORD *)(a3 + 120));
    v24 = *(_DWORD *)v26;
    v14 = sub_BD5D20(a2);
    v16 = v15;
    v17 = v14;
    v18 = strlen(v11);
    v19 = sub_ADD600((int)a1 + 16, v22, (_DWORD)v11, v18, v17, v16, v21, v24, v23, v25, 1, 0, 0, 0, 0, 0);
    return (const char *)sub_B996C0(a2, v19);
  }
  return result;
}
