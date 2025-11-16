// Function: sub_15B9E00
// Address: 0x15b9e00
//
__int64 __fastcall sub_15B9E00(__int64 *a1, int a2, unsigned int a3, __int64 a4, __int64 a5, unsigned int a6, char a7)
{
  int v7; // r10d
  __int64 v8; // r11
  unsigned int v12; // ebx
  char v13; // r9
  __int64 v14; // r15
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdi
  int v21; // eax
  unsigned int v22; // edi
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r8
  int v26; // [rsp+0h] [rbp-80h]
  unsigned int v27; // [rsp+4h] [rbp-7Ch]
  int v29; // [rsp+18h] [rbp-68h]
  int v30; // [rsp+1Ch] [rbp-64h]
  int v31; // [rsp+1Ch] [rbp-64h]
  unsigned int v32; // [rsp+1Ch] [rbp-64h]
  __int64 v33; // [rsp+20h] [rbp-60h]
  __int64 v34; // [rsp+20h] [rbp-60h]
  __int64 v35; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+28h] [rbp-58h]
  __int64 *v37; // [rsp+30h] [rbp-50h] BYREF
  __int64 v38; // [rsp+38h] [rbp-48h] BYREF
  __int64 v39[8]; // [rsp+40h] [rbp-40h] BYREF

  v7 = a2;
  v8 = a4;
  v12 = a3;
  v13 = a7;
  if ( a3 >= 0x10000 )
    v12 = 0;
  if ( a6 )
  {
LABEL_6:
    v39[0] = v8;
    v37 = v39;
    v38 = 0x200000001LL;
    if ( a5 )
    {
      v39[1] = a5;
      v16 = 2;
      v17 = 2;
      LODWORD(v38) = 2;
    }
    else
    {
      v16 = 1;
      v17 = 1;
    }
    v31 = v7;
    v36 = v16;
    v18 = *a1 + 528;
    v19 = sub_161E980(24, v17);
    v20 = v19;
    if ( v19 )
    {
      v34 = v19;
      sub_15AFCF0(v19, (int)a1, a6, v31, v12, v36, (__int64)v39, v36);
      v20 = v34;
    }
    result = sub_15B9C60(v20, a6, v18);
    if ( v37 != v39 )
    {
      v35 = result;
      _libc_free((unsigned __int64)v37);
      return v35;
    }
    return result;
  }
  v14 = *a1;
  v37 = (__int64 *)__PAIR64__(v12, a2);
  v38 = a4;
  v39[0] = a5;
  v30 = *(_DWORD *)(v14 + 552);
  v33 = *(_QWORD *)(v14 + 536);
  if ( !v30 )
    goto LABEL_5;
  v21 = sub_15B3100((int *)&v37, (int *)&v37 + 1, &v38, v39);
  v7 = a2;
  v8 = a4;
  v13 = a7;
  v22 = (v30 - 1) & v21;
  v23 = (__int64 *)(v33 + 8LL * v22);
  v24 = *v23;
  if ( *v23 == -8 )
    goto LABEL_5;
  v29 = 1;
  v26 = v30 - 1;
  v27 = v22;
  while ( 1 )
  {
    if ( v24 != -16 && v37 == (__int64 *)__PAIR64__(*(unsigned __int16 *)(v24 + 2), *(_DWORD *)(v24 + 4)) )
    {
      v32 = *(_DWORD *)(v24 + 8);
      if ( v38 == *(_QWORD *)(v24 - 8LL * v32) )
      {
        v25 = 0;
        if ( v32 == 2 )
          v25 = *(_QWORD *)(v24 - 8);
        if ( v39[0] == v25 )
          break;
      }
    }
    v27 = v26 & (v29 + v27);
    v23 = (__int64 *)(v33 + 8LL * v27);
    v24 = *v23;
    if ( *v23 == -8 )
      goto LABEL_5;
    ++v29;
  }
  if ( v23 == (__int64 *)(*(_QWORD *)(v14 + 536) + 8LL * *(unsigned int *)(v14 + 552)) || (result = *v23) == 0 )
  {
LABEL_5:
    result = 0;
    if ( !v13 )
      return result;
    goto LABEL_6;
  }
  return result;
}
