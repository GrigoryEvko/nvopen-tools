// Function: sub_28626B0
// Address: 0x28626b0
//
__int64 __fastcall sub_28626B0(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r13d
  int v6; // r13d
  _QWORD *v7; // r15
  size_t v8; // r14
  int v9; // eax
  __int64 v10; // r8
  int v11; // r11d
  __int64 v12; // r10
  unsigned int i; // ecx
  __int64 v14; // r12
  __int64 v15; // r9
  unsigned int v16; // eax
  int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-B8h]
  __int64 v19; // [rsp+10h] [rbp-B0h]
  int v20; // [rsp+18h] [rbp-A8h]
  unsigned int v21; // [rsp+1Ch] [rbp-A4h]
  __int64 v22; // [rsp+20h] [rbp-A0h]
  __int64 v23; // [rsp+20h] [rbp-A0h]
  __int64 v24; // [rsp+28h] [rbp-98h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v4 - 1;
  v7 = *(_QWORD **)a2;
  v8 = 8LL * *(unsigned int *)(a2 + 8);
  v24 = *(_QWORD *)(a1 + 8);
  v22 = *(unsigned int *)(a2 + 8);
  v9 = sub_2862070(v7, (__int64)&v7[v8 / 8]);
  v10 = v22;
  v11 = 1;
  v12 = 0;
  for ( i = v6 & v9; ; i = v6 & v16 )
  {
    v14 = v24 + 48LL * i;
    v15 = *(unsigned int *)(v14 + 8);
    if ( v10 == v15 )
    {
      if ( !v8 )
        goto LABEL_9;
      v21 = i;
      v18 = *(unsigned int *)(v14 + 8);
      v19 = v10;
      v20 = v11;
      v23 = v12;
      v17 = memcmp(v7, *(const void **)v14, v8);
      v12 = v23;
      i = v21;
      v11 = v20;
      v10 = v19;
      v15 = v18;
      if ( !v17 )
      {
LABEL_9:
        *a3 = v14;
        return 1;
      }
    }
    if ( v15 == 1 )
      break;
LABEL_6:
    v16 = i + v11++;
  }
  if ( **(_QWORD **)v14 != -1 )
  {
    if ( **(_QWORD **)v14 == -2 && !v12 )
      v12 = v14;
    goto LABEL_6;
  }
  if ( !v12 )
    v12 = v14;
  *a3 = v12;
  return 0;
}
