// Function: sub_B09060
// Address: 0xb09060
//
__int64 __fastcall sub_B09060(__int64 *a1, __int64 a2, __int64 a3, int a4, unsigned int a5, char a6)
{
  __int64 v6; // r11
  __int64 v7; // r15
  __int64 *v9; // r13
  unsigned int v10; // r12d
  __int64 v11; // r8
  int v12; // ebx
  int v13; // ebx
  int v14; // r13d
  unsigned int i; // r12d
  __int64 v16; // r15
  _BYTE *v17; // rax
  unsigned int v18; // ecx
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // r11
  __int64 v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // r10
  __int64 v28; // rbx
  __int64 v29; // [rsp+0h] [rbp-90h]
  __int64 v31; // [rsp+28h] [rbp-68h]
  __int64 v32; // [rsp+30h] [rbp-60h]
  __int64 v34; // [rsp+40h] [rbp-50h] BYREF
  __int64 v35; // [rsp+48h] [rbp-48h] BYREF
  int v36[16]; // [rsp+50h] [rbp-40h] BYREF

  v6 = a3;
  v7 = a2;
  v9 = a1;
  v10 = a5;
  if ( a5 )
    goto LABEL_10;
  v11 = *a1;
  v34 = a2;
  v35 = a3;
  v36[0] = a4;
  v12 = *(_DWORD *)(v11 + 1136);
  v32 = *(_QWORD *)(v11 + 1120);
  if ( v12 )
  {
    v13 = v12 - 1;
    v29 = v11;
    v14 = 1;
    for ( i = v13 & sub_AF7750(&v34, &v35, v36); ; i = v13 & v18 )
    {
      v16 = *(_QWORD *)(v32 + 8LL * i);
      if ( v16 == -4096 )
      {
        v9 = a1;
        v7 = a2;
        v6 = a3;
        v10 = 0;
        goto LABEL_9;
      }
      if ( v16 != -8192 )
      {
        v17 = sub_A17150((_BYTE *)(v16 - 16));
        if ( v34 == *((_QWORD *)v17 + 1) )
        {
          v24 = v35;
          v25 = v16;
          if ( *(_BYTE *)v16 != 16 )
          {
            v31 = v35;
            v26 = sub_A17150((_BYTE *)(v16 - 16));
            v24 = v31;
            v25 = *(_QWORD *)v26;
          }
          if ( v24 == v25 && v36[0] == *(_DWORD *)(v16 + 4) )
            break;
        }
      }
      v18 = i + v14++;
    }
    v27 = v32 + 8LL * i;
    v28 = v16;
    v9 = a1;
    v6 = a3;
    v10 = 0;
    v7 = a2;
    if ( v27 != *(_QWORD *)(v29 + 1120) + 8LL * *(unsigned int *)(v29 + 1136) )
      return v28;
  }
LABEL_9:
  result = 0;
  if ( a6 )
  {
LABEL_10:
    v20 = *v9;
    v35 = v7;
    v34 = v6;
    v21 = v20 + 1112;
    v22 = sub_B97910(16, 2, v10);
    v23 = v22;
    if ( v22 )
    {
      sub_AF3E20(v22, (int)v9, 20, v10, (int)&v34, 2);
      *(_DWORD *)(v23 + 4) = a4;
    }
    return sub_B08E90(v23, v10, v21);
  }
  return result;
}
