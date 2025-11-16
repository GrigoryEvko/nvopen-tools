// Function: sub_B089A0
// Address: 0xb089a0
//
__int64 __fastcall sub_B089A0(__int64 *a1, __int64 a2, __int64 a3, int a4, unsigned int a5, unsigned int a6, char a7)
{
  __int64 v7; // r10
  __int64 v8; // r11
  int v12; // ebx
  __int64 v13; // r8
  int v14; // r14d
  int v15; // eax
  int v17; // r9d
  int v18; // ecx
  unsigned int v19; // esi
  __int64 *v20; // r8
  __int64 v21; // r14
  unsigned __int8 v22; // al
  __int64 v23; // rdi
  __int64 v24; // rax
  _BYTE *v25; // rax
  unsigned int v26; // esi
  __int64 result; // rax
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // [rsp+10h] [rbp-90h]
  int v33; // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+20h] [rbp-80h]
  int v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+30h] [rbp-70h]
  __int64 v38; // [rsp+38h] [rbp-68h]
  __int64 v39; // [rsp+38h] [rbp-68h]
  __int64 v40; // [rsp+40h] [rbp-60h]
  __int64 v41; // [rsp+48h] [rbp-58h]
  __int64 v42; // [rsp+50h] [rbp-50h] BYREF
  __int64 v43; // [rsp+58h] [rbp-48h] BYREF
  int v44; // [rsp+60h] [rbp-40h] BYREF
  int v45[15]; // [rsp+64h] [rbp-3Ch] BYREF

  v7 = a2;
  v8 = a3;
  v12 = a5;
  if ( a5 >= 0x10000 )
    v12 = 0;
  if ( a6 )
    goto LABEL_17;
  v13 = *a1;
  v42 = a2;
  v43 = a3;
  v44 = a4;
  v45[0] = v12;
  v14 = *(_DWORD *)(v13 + 1104);
  v38 = v13;
  v40 = *(_QWORD *)(v13 + 1088);
  if ( v14 )
  {
    v15 = sub_AF7510(&v42, &v43, &v44, v45);
    v17 = v14 - 1;
    v8 = a3;
    v18 = 1;
    v34 = v38;
    v19 = (v14 - 1) & v15;
    v39 = a2;
    while ( 1 )
    {
      v20 = (__int64 *)(v40 + 8LL * v19);
      v21 = *v20;
      if ( *v20 == -4096 )
      {
        v7 = v39;
        goto LABEL_16;
      }
      if ( v21 != -8192 )
      {
        v22 = *(_BYTE *)(v21 - 16);
        v23 = (v22 & 2) != 0 ? *(_QWORD *)(v21 - 32) : v21 - 16 - 8LL * ((v22 >> 2) & 0xF);
        if ( v42 == *(_QWORD *)(v23 + 8) )
        {
          v37 = v43;
          v24 = *v20;
          if ( *(_BYTE *)v21 != 16 )
          {
            v32 = v8;
            v33 = v18;
            v36 = v17;
            v25 = sub_A17150((_BYTE *)(v21 - 16));
            v20 = (__int64 *)(v40 + 8LL * v19);
            v17 = v36;
            v24 = *(_QWORD *)v25;
            v8 = v32;
            v18 = v33;
          }
          if ( v37 == v24 && v44 == *(_DWORD *)(v21 + 4) && v45[0] == *(unsigned __int16 *)(v21 + 16) )
            break;
        }
      }
      v26 = v18 + v19;
      ++v18;
      v19 = v17 & v26;
    }
    v7 = v39;
    if ( v20 != (__int64 *)(*(_QWORD *)(v34 + 1088) + 8LL * *(unsigned int *)(v34 + 1104)) )
      return v21;
  }
LABEL_16:
  result = 0;
  if ( a7 )
  {
LABEL_17:
    v28 = *a1;
    v42 = v8;
    v43 = v7;
    v29 = v28 + 1080;
    v30 = sub_B97910(24, 2, a6);
    v31 = v30;
    if ( v30 )
    {
      v41 = v30;
      sub_AF3E20(v30, (int)a1, 19, a6, (int)&v42, 2);
      v31 = v41;
      *(_WORD *)(v41 + 16) = v12;
      *(_DWORD *)(v41 + 4) = a4;
    }
    return sub_B087A0(v31, a6, v29);
  }
  return result;
}
