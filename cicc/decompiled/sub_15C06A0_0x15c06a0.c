// Function: sub_15C06A0
// Address: 0x15c06a0
//
__int64 __fastcall sub_15C06A0(__int64 *a1, __int64 a2, __int64 a3, int a4, unsigned int a5, unsigned int a6, char a7)
{
  __int64 v7; // r10
  int v11; // ebx
  char v12; // r11
  __int64 v13; // r8
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdi
  int v19; // eax
  int v20; // r9d
  unsigned int v21; // edi
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // [rsp+4h] [rbp-7Ch]
  __int64 v27; // [rsp+18h] [rbp-68h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  int v29; // [rsp+20h] [rbp-60h]
  __int64 v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+28h] [rbp-58h]
  __int64 v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34; // [rsp+38h] [rbp-48h] BYREF
  int v35; // [rsp+40h] [rbp-40h] BYREF
  int v36[15]; // [rsp+44h] [rbp-3Ch] BYREF

  v7 = a3;
  v11 = a5;
  v12 = a7;
  if ( a5 >= 0x10000 )
    v11 = 0;
  if ( a6 )
  {
LABEL_6:
    v15 = *a1;
    v34 = a2;
    v33 = v7;
    v16 = v15 + 912;
    v17 = sub_161E980(32, 2);
    v18 = v17;
    if ( v17 )
    {
      v32 = v17;
      sub_1623D80(v17, (_DWORD)a1, 18, a6, (unsigned int)&v33, 2, 0, 0);
      v18 = v32;
      *(_WORD *)(v32 + 2) = 11;
      *(_DWORD *)(v32 + 24) = a4;
      *(_WORD *)(v32 + 28) = v11;
    }
    return sub_15C04F0(v18, a6, v16);
  }
  v13 = *a1;
  v33 = a2;
  v34 = a3;
  v35 = a4;
  v36[0] = v11;
  v27 = v13;
  v29 = *(_DWORD *)(v13 + 936);
  v31 = *(_QWORD *)(v13 + 920);
  if ( !v29 )
    goto LABEL_5;
  v19 = sub_15B2700(&v33, &v34, &v35, v36);
  v7 = a3;
  v12 = a7;
  v20 = v29 - 1;
  v21 = (v29 - 1) & v19;
  v22 = (__int64 *)(v31 + 8LL * v21);
  v23 = *v22;
  if ( *v22 == -8 )
    goto LABEL_5;
  v25 = 1;
  v30 = v27;
  while ( 1 )
  {
    if ( v23 != -16 )
    {
      v24 = *(unsigned int *)(v23 + 8);
      v28 = v23;
      if ( v33 == *(_QWORD *)(v23 + 8 * (1 - v24)) )
      {
        if ( *(_BYTE *)v23 != 15 )
          v28 = *(_QWORD *)(v23 - 8 * v24);
        if ( v34 == v28 && v35 == *(_DWORD *)(v23 + 24) && v36[0] == *(unsigned __int16 *)(v23 + 28) )
          break;
      }
    }
    v21 = v20 & (v25 + v21);
    v22 = (__int64 *)(v31 + 8LL * v21);
    v23 = *v22;
    if ( *v22 == -8 )
      goto LABEL_5;
    ++v25;
  }
  if ( v22 == (__int64 *)(*(_QWORD *)(v30 + 920) + 8LL * *(unsigned int *)(v30 + 936)) || (result = *v22) == 0 )
  {
LABEL_5:
    result = 0;
    if ( !v12 )
      return result;
    goto LABEL_6;
  }
  return result;
}
