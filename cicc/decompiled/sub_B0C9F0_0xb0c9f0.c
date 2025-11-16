// Function: sub_B0C9F0
// Address: 0xb0c9f0
//
__int64 __fastcall sub_B0C9F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned int a6, char a7)
{
  __int64 v7; // r10
  unsigned int v11; // r12d
  __int64 v12; // rax
  int v13; // ebx
  int v14; // eax
  int v15; // r8d
  unsigned int v16; // r10d
  int v17; // r11d
  unsigned int i; // r9d
  __int64 *v19; // r12
  __int64 v20; // rbx
  unsigned __int8 v21; // al
  _QWORD *v22; // rdi
  __int64 v23; // rax
  unsigned int v24; // r9d
  __int64 result; // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // r15
  _BYTE *v30; // rax
  __int64 *v31; // rsi
  unsigned int v32; // [rsp+Ch] [rbp-94h]
  __int64 v33; // [rsp+10h] [rbp-90h]
  int v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+20h] [rbp-80h]
  int v36; // [rsp+24h] [rbp-7Ch]
  __int64 v37; // [rsp+28h] [rbp-78h]
  __int64 v38; // [rsp+30h] [rbp-70h]
  __int64 v40; // [rsp+40h] [rbp-60h]
  __int64 v41; // [rsp+50h] [rbp-50h] BYREF
  __int64 v42; // [rsp+58h] [rbp-48h] BYREF
  __int64 v43; // [rsp+60h] [rbp-40h]
  int v44[14]; // [rsp+68h] [rbp-38h] BYREF

  v7 = a3;
  v11 = a6;
  if ( a6 )
    goto LABEL_13;
  v12 = *a1;
  v41 = a2;
  v42 = a3;
  v43 = a4;
  v44[0] = a5;
  v37 = v12;
  v40 = *(_QWORD *)(v12 + 1344);
  v13 = *(_DWORD *)(v12 + 1360);
  if ( v13 )
  {
    v38 = a4;
    v14 = sub_AF8830(&v41, &v42, v44);
    v15 = v13 - 1;
    a4 = v38;
    v16 = 0;
    v17 = 1;
    for ( i = (v13 - 1) & v14; ; i = v15 & v24 )
    {
      v19 = (__int64 *)(v40 + 8LL * i);
      v20 = *v19;
      if ( *v19 == -4096 )
      {
        v11 = v16;
        v7 = a3;
        goto LABEL_12;
      }
      if ( v20 != -8192 )
      {
        v21 = *(_BYTE *)(v20 - 16);
        v22 = (v21 & 2) != 0 ? *(_QWORD **)(v20 - 32) : (_QWORD *)(v20 - 16 - 8LL * ((v21 >> 2) & 0xF));
        if ( v41 == *v22 )
        {
          v32 = v16;
          v33 = a4;
          v34 = v17;
          v35 = i;
          v36 = v15;
          v23 = sub_AF5140(*v19, 1u);
          v15 = v36;
          i = v35;
          v17 = v34;
          a4 = v33;
          v16 = v32;
          if ( v42 == v23 )
          {
            v30 = sub_A17150((_BYTE *)(v20 - 16));
            v15 = v36;
            i = v35;
            v17 = v34;
            a4 = v33;
            v16 = v32;
            if ( v43 == *((_QWORD *)v30 + 2) && v44[0] == *(_DWORD *)(v20 + 4) )
              break;
          }
        }
      }
      v24 = v17 + i;
      ++v17;
    }
    v31 = v19;
    v11 = v32;
    v7 = a3;
    if ( v31 != (__int64 *)(*(_QWORD *)(v37 + 1344) + 8LL * *(unsigned int *)(v37 + 1360)) )
      return v20;
  }
LABEL_12:
  result = 0;
  if ( a7 )
  {
LABEL_13:
    v26 = *a1;
    v41 = a2;
    v27 = v26 + 1336;
    v42 = v7;
    v43 = a4;
    v28 = sub_B97910(16, 3, v11);
    v29 = v28;
    if ( v28 )
      sub_AF40A0(v28, (int)a1, v11, a5, (int)&v41, 3);
    return sub_B0C7D0(v29, v11, v27);
  }
  return result;
}
