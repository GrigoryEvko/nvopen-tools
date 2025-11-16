// Function: sub_B0F520
// Address: 0xb0f520
//
__int64 __fastcall sub_B0F520(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        unsigned int a9,
        char a10)
{
  __int64 v10; // r10
  __int64 v11; // r14
  int v12; // r13d
  __int64 *v13; // r12
  unsigned int v15; // r15d
  __int64 v16; // rax
  int v17; // r13d
  unsigned int i; // r12d
  __int64 *v19; // r14
  __int64 v20; // r15
  __int64 v21; // rax
  unsigned int v22; // r11d
  __int64 result; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // r9
  __int64 v28; // rdi
  _BYTE *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  _BYTE *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rcx
  __int64 v36; // [rsp+18h] [rbp-A8h]
  int v38; // [rsp+28h] [rbp-98h]
  __int64 v39; // [rsp+30h] [rbp-90h]
  __int64 v42; // [rsp+40h] [rbp-80h]
  __int64 v43; // [rsp+50h] [rbp-70h] BYREF
  __int64 v44; // [rsp+58h] [rbp-68h] BYREF
  __int64 v45; // [rsp+60h] [rbp-60h] BYREF
  __int64 v46; // [rsp+68h] [rbp-58h] BYREF
  __int64 v47; // [rsp+70h] [rbp-50h] BYREF
  int v48; // [rsp+78h] [rbp-48h] BYREF
  __int64 v49[8]; // [rsp+80h] [rbp-40h] BYREF

  v10 = a6;
  v11 = a5;
  v12 = a4;
  v13 = a1;
  v15 = a9;
  if ( a9 )
    goto LABEL_10;
  v16 = *a1;
  v44 = a3;
  LODWORD(v45) = a4;
  v43 = a2;
  v46 = a5;
  v47 = a6;
  v48 = a7;
  v49[0] = a8;
  v36 = v16;
  v38 = *(_DWORD *)(v16 + 1392);
  v39 = *(_QWORD *)(v16 + 1376);
  if ( v38 )
  {
    v17 = 1;
    for ( i = (v38 - 1) & sub_AF9E80(&v43, &v44, (int *)&v45, &v46, &v47, &v48, v49); ; i = (v38 - 1) & v22 )
    {
      v19 = (__int64 *)(v39 + 8LL * i);
      v20 = *v19;
      if ( *v19 == -4096 )
      {
        v13 = a1;
        v12 = a4;
        v11 = a5;
        v10 = a6;
        v15 = 0;
        goto LABEL_9;
      }
      if ( v20 != -8192 )
      {
        v21 = sub_AF5140(*v19, 0);
        if ( v43 == v21 )
        {
          v29 = sub_A17150((_BYTE *)(v20 - 16));
          if ( v44 == *((_QWORD *)v29 + 1) && (_DWORD)v45 == *(_DWORD *)(v20 + 16) )
          {
            v30 = sub_AF5140(v20, 2u);
            if ( v46 == v30 )
            {
              v31 = sub_AF5140(v20, 3u);
              if ( v47 == v31 && v48 == *(_DWORD *)(v20 + 20) )
              {
                v32 = sub_A17150((_BYTE *)(v20 - 16));
                if ( v49[0] == *((_QWORD *)v32 + 4) )
                  break;
              }
            }
          }
        }
      }
      v22 = i + v17++;
    }
    v33 = v39 + 8LL * i;
    v34 = v20;
    v13 = a1;
    v12 = a4;
    v11 = a5;
    v10 = a6;
    v15 = 0;
    if ( v33 != *(_QWORD *)(v36 + 1376) + 8LL * *(unsigned int *)(v36 + 1392) )
      return v34;
  }
LABEL_9:
  result = 0;
  if ( a10 )
  {
LABEL_10:
    v44 = a3;
    v45 = v11;
    v43 = a2;
    v47 = a8;
    v24 = *v13;
    v46 = v10;
    v25 = v24 + 1368;
    v26 = sub_B97910(24, 5, v15);
    v28 = v26;
    if ( v26 )
    {
      v42 = v26;
      sub_AF5010(v26, (int)v13, v15, v12, a7, v27, (__int64)&v43, 5);
      v28 = v42;
    }
    return sub_B0F440(v28, v15, v25);
  }
  return result;
}
