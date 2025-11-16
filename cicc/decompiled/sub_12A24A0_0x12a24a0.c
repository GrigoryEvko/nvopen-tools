// Function: sub_12A24A0
// Address: 0x12a24a0
//
const char *__fastcall sub_12A24A0(_QWORD *a1, __int64 a2, __int64 a3, const char *a4)
{
  const char *v5; // r12
  const char *result; // rax
  const char *v7; // r12
  __int64 v8; // r9
  __int64 v9; // r10
  int v10; // eax
  int v11; // edx
  int v12; // r15d
  int v13; // r14d
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-60h]
  int v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  int v20; // [rsp+18h] [rbp-48h]
  char v21; // [rsp+1Fh] [rbp-41h]
  char v22[52]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = a4;
  if ( !a4 )
  {
    v5 = *(const char **)(a3 + 8);
    if ( !v5 )
      v5 = byte_3F871B3;
  }
  sub_129E300(*(_DWORD *)(a3 + 64), v22);
  result = (const char *)sub_129E180((__int64)v5, a3);
  v7 = result;
  if ( *result )
  {
    v8 = sub_129F850((__int64)a1, *(_DWORD *)(a3 + 64));
    v21 = *(_BYTE *)(a3 + 136) == 2;
    if ( *(_BYTE *)(a3 + 136) != 2 || (v16 = a1[68], v16 == a1[64]) )
    {
      v9 = a1[1];
    }
    else
    {
      if ( v16 == a1[69] )
        v16 = *(_QWORD *)(a1[71] - 8LL) + 512LL;
      v9 = *(_QWORD *)(v16 - 8);
    }
    v17 = v8;
    v18 = v9;
    v19 = sub_12A0C10((__int64)a1, *(_QWORD *)(a3 + 120));
    v20 = *(_DWORD *)v22;
    v10 = sub_1649960(a2);
    v12 = v11;
    v13 = v10;
    v14 = strlen(v7);
    v15 = sub_15A6890((int)a1 + 16, v18, (_DWORD)v7, v14, v13, v12, v17, v20, v19, v21, 0, 0, 0);
    return (const char *)sub_1626A90(a2, v15);
  }
  return result;
}
