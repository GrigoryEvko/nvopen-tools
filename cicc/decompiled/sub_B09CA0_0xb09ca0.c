// Function: sub_B09CA0
// Address: 0xb09ca0
//
__int64 __fastcall sub_B09CA0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        unsigned int a7,
        char a8)
{
  __int64 v8; // r10
  __int64 v9; // r11
  __int64 v10; // r14
  int v11; // r13d
  __int64 *v12; // r12
  __int64 v13; // rbx
  unsigned int v14; // r15d
  __int64 v15; // rax
  int v16; // eax
  unsigned int v17; // r11d
  int v18; // r10d
  int v20; // r9d
  unsigned int v21; // r8d
  __int64 v22; // rcx
  __int64 *v23; // r12
  __int64 v24; // rbx
  unsigned __int8 v25; // al
  _QWORD *v26; // rsi
  _BYTE *v27; // rax
  unsigned int v28; // r8d
  __int64 result; // rax
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // rax
  _BYTE *v35; // rax
  __int64 *v36; // rsi
  __int64 v37; // rcx
  __int64 v38; // [rsp+0h] [rbp-A0h]
  int v39; // [rsp+8h] [rbp-98h]
  int v40; // [rsp+Ch] [rbp-94h]
  __int64 v41; // [rsp+10h] [rbp-90h]
  unsigned int v44; // [rsp+28h] [rbp-78h]
  int v45; // [rsp+2Ch] [rbp-74h]
  int v46; // [rsp+2Ch] [rbp-74h]
  __int64 v47; // [rsp+30h] [rbp-70h]
  __int64 v48; // [rsp+40h] [rbp-60h] BYREF
  __int64 v49; // [rsp+48h] [rbp-58h] BYREF
  __int64 v50; // [rsp+50h] [rbp-50h] BYREF
  __int64 v51; // [rsp+58h] [rbp-48h] BYREF
  int v52[16]; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v53; // [rsp+B0h] [rbp+10h]

  v8 = a4;
  v9 = a5;
  v10 = a3;
  v11 = a6;
  v12 = a1;
  v13 = a2;
  v14 = a7;
  if ( a7 )
    goto LABEL_13;
  v15 = *a1;
  v48 = a2;
  v49 = a3;
  v50 = a4;
  v51 = a5;
  v52[0] = a6;
  v41 = v15;
  v45 = *(_DWORD *)(v15 + 1520);
  v47 = *(_QWORD *)(v15 + 1504);
  if ( v45 )
  {
    v16 = sub_AF9890(&v48, &v49, &v50, &v51, v52);
    v17 = 0;
    v18 = v11;
    v20 = 1;
    v46 = v45 - 1;
    v21 = v46 & v16;
    v22 = v10;
    while ( 1 )
    {
      v23 = (__int64 *)(v47 + 8LL * v21);
      v24 = *v23;
      if ( *v23 == -4096 )
      {
        v13 = a2;
        v11 = v18;
        v14 = v17;
        v8 = a4;
        v9 = a5;
        v12 = a1;
        v10 = v22;
        goto LABEL_12;
      }
      if ( v24 != -8192 )
      {
        v25 = *(_BYTE *)(v24 - 16);
        v26 = (v25 & 2) != 0 ? *(_QWORD **)(v24 - 32) : (_QWORD *)(v24 - 16 - 8LL * ((v25 >> 2) & 0xF));
        if ( v48 == *v26 )
        {
          v53 = v17;
          v39 = v18;
          v38 = v22;
          v40 = v20;
          v44 = v21;
          v27 = sub_A17150((_BYTE *)(v24 - 16));
          v21 = v44;
          v17 = v53;
          v20 = v40;
          v22 = v38;
          v18 = v39;
          if ( v49 == *((_QWORD *)v27 + 1) )
          {
            v34 = sub_AF5140(v24, 2u);
            v21 = v44;
            v20 = v40;
            v22 = v38;
            v18 = v39;
            v17 = v53;
            if ( v50 == v34 )
            {
              v35 = sub_A17150((_BYTE *)(v24 - 16));
              v21 = v44;
              v17 = v53;
              v20 = v40;
              v22 = v38;
              v18 = v39;
              if ( v51 == *((_QWORD *)v35 + 3) && v52[0] == *(_DWORD *)(v24 + 4) )
                break;
            }
          }
        }
      }
      v28 = v20 + v21;
      ++v20;
      v21 = v46 & v28;
    }
    v36 = v23;
    v11 = v39;
    v12 = a1;
    v8 = a4;
    v10 = v38;
    v37 = v24;
    v13 = a2;
    v14 = v53;
    v9 = a5;
    if ( v36 != (__int64 *)(*(_QWORD *)(v41 + 1504) + 8LL * *(unsigned int *)(v41 + 1520)) )
      return v37;
  }
LABEL_12:
  result = 0;
  if ( a8 )
  {
LABEL_13:
    v30 = *v12;
    v48 = v13;
    v49 = v10;
    v31 = v30 + 1496;
    v50 = v8;
    v51 = v9;
    v32 = sub_B97910(16, 4, v14);
    v33 = v32;
    if ( v32 )
      sub_AF3EA0(v32, (int)v12, v14, v11, (int)&v48, 4);
    return sub_B09BC0(v33, v14, v31);
  }
  return result;
}
