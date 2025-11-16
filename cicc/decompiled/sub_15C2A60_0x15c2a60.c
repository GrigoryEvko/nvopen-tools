// Function: sub_15C2A60
// Address: 0x15c2a60
//
__int64 __fastcall sub_15C2A60(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6, char a7)
{
  char v11; // r11
  __int64 v12; // r9
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r14
  int v18; // eax
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r9
  int v23; // [rsp+4h] [rbp-7Ch]
  __int64 v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  int v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h] BYREF
  __int64 v31; // [rsp+40h] [rbp-40h] BYREF
  __int64 v32[7]; // [rsp+48h] [rbp-38h] BYREF

  v11 = a7;
  if ( a6 )
  {
LABEL_4:
    v14 = *a1;
    v29 = a3;
    v30 = a4;
    v31 = a5;
    v15 = v14 + 1072;
    v16 = sub_161E980(24, 3);
    v17 = v16;
    if ( v16 )
    {
      sub_1623D80(v16, (_DWORD)a1, 23, a6, (unsigned int)&v29, 3, 0, 0);
      *(_WORD *)(v17 + 2) = a2;
    }
    return sub_15C28B0(v17, a6, v15);
  }
  v12 = *a1;
  LODWORD(v29) = a2;
  v30 = a3;
  v31 = a4;
  v32[0] = a5;
  v25 = v12;
  v27 = *(_DWORD *)(v12 + 1096);
  v28 = *(_QWORD *)(v12 + 1080);
  if ( !v27 )
    goto LABEL_3;
  v24 = a5;
  v18 = sub_15B49E0((__int32 *)&v29, &v30, &v31, v32);
  a5 = v24;
  v11 = a7;
  v19 = (v27 - 1) & v18;
  v20 = (__int64 *)(v28 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == -8 )
    goto LABEL_3;
  v23 = 1;
  v22 = v25;
  while ( 1 )
  {
    if ( v21 != -16 && (_DWORD)v29 == *(unsigned __int16 *)(v21 + 2) )
    {
      v26 = *(unsigned int *)(v21 + 8);
      if ( v30 == *(_QWORD *)(v21 - 8 * v26)
        && v31 == *(_QWORD *)(v21 + 8 * (1 - v26))
        && v32[0] == *(_QWORD *)(v21 + 8 * (2 - v26)) )
      {
        break;
      }
    }
    v19 = (v27 - 1) & (v23 + v19);
    v20 = (__int64 *)(v28 + 8LL * v19);
    v21 = *v20;
    if ( *v20 == -8 )
      goto LABEL_3;
    ++v23;
  }
  if ( v20 == (__int64 *)(*(_QWORD *)(v22 + 1080) + 8LL * *(unsigned int *)(v22 + 1096)) || (result = *v20) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !v11 )
      return result;
    goto LABEL_4;
  }
  return result;
}
