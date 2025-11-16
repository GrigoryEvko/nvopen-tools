// Function: sub_B0AC10
// Address: 0xb0ac10
//
__int64 __fastcall sub_B0AC10(__int64 *a1, __int64 a2, __int64 a3, __int8 a4, unsigned int a5, char a6)
{
  __int64 v6; // r11
  __int64 v7; // r15
  __int64 *v8; // r13
  unsigned int v9; // r12d
  char v10; // bl
  __int64 v11; // r8
  int v12; // r14d
  int v13; // ebx
  int v14; // r13d
  unsigned int i; // r12d
  __int64 *v16; // r14
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned int v19; // ecx
  __int64 result; // rax
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r14
  _BYTE *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r14
  __int64 v28; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+30h] [rbp-60h]
  __int64 v34; // [rsp+40h] [rbp-50h] BYREF
  __int64 v35; // [rsp+48h] [rbp-48h] BYREF
  __int8 v36[64]; // [rsp+50h] [rbp-40h] BYREF

  v6 = a3;
  v7 = a2;
  v8 = a1;
  v9 = a5;
  v10 = a4;
  if ( a5 )
    goto LABEL_10;
  v11 = *a1;
  v34 = a2;
  v35 = a3;
  v36[0] = a4;
  v12 = *(_DWORD *)(v11 + 1232);
  v32 = *(_QWORD *)(v11 + 1216);
  if ( v12 )
  {
    v13 = v12 - 1;
    v28 = v11;
    v14 = 1;
    for ( i = (v12 - 1) & sub_AFA150(&v34, &v35, v36); ; i = v13 & v19 )
    {
      v16 = (__int64 *)(v32 + 8LL * i);
      v17 = *v16;
      if ( *v16 == -4096 )
      {
        v8 = a1;
        v7 = a2;
        v6 = a3;
        v10 = a4;
        v9 = 0;
        goto LABEL_9;
      }
      if ( v17 != -8192 )
      {
        v18 = sub_AF5140(*v16, 0);
        if ( v34 == v18 )
        {
          v25 = sub_A17150((_BYTE *)(v17 - 16));
          if ( v35 == *((_QWORD *)v25 + 1) && v36[0] == *(_BYTE *)(v17 + 1) >> 7 )
            break;
        }
      }
      v19 = i + v14++;
    }
    v26 = v32 + 8LL * i;
    v8 = a1;
    v27 = v17;
    v6 = a3;
    v7 = a2;
    v10 = a4;
    v9 = 0;
    if ( v26 != *(_QWORD *)(v28 + 1216) + 8LL * *(unsigned int *)(v28 + 1232) )
      return v27;
  }
LABEL_9:
  result = 0;
  if ( a6 )
  {
LABEL_10:
    v21 = *v8;
    v34 = v7;
    v22 = v21 + 1208;
    v35 = v6;
    v23 = sub_B97910(16, 2, v9);
    v24 = v23;
    if ( v23 )
      sub_AF3F40(v23, (int)v8, v9, v10, (int)&v34, 2);
    return sub_B0AA50(v24, v9, v22);
  }
  return result;
}
