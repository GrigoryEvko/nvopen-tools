// Function: sub_3449EC0
// Address: 0x3449ec0
//
bool __fastcall sub_3449EC0(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // r13
  __int16 *v10; // rdx
  __int16 v11; // ax
  __int64 v12; // rdx
  unsigned __int16 v13; // cx
  int v14; // eax
  __int64 v15; // rdi
  unsigned int v16; // ebx
  _QWORD *v17; // rax
  char v18; // bl
  __int16 v19; // [rsp-38h] [rbp-38h] BYREF
  __int64 v20; // [rsp-30h] [rbp-30h]

  if ( !a2 )
    return 0;
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 == 35 || v7 == 11 )
  {
    v8 = a2;
  }
  else
  {
    if ( v7 != 156 )
      return 0;
    v8 = sub_33E1640(a2, 0, a3, a4, a5, a6);
    if ( !v8 )
      return 0;
  }
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v19 = v11;
  v20 = v12;
  if ( v11 )
  {
    v13 = v11 - 17;
    if ( (unsigned __int16)(v11 - 10) > 6u && (unsigned __int16)(v11 - 126) > 0x31u )
    {
      if ( v13 > 0xD3u )
      {
LABEL_12:
        v14 = a1[15];
        goto LABEL_13;
      }
LABEL_23:
      v14 = a1[17];
LABEL_13:
      v15 = *(_QWORD *)(v8 + 96);
      v16 = *(_DWORD *)(v15 + 32);
      if ( !v14 )
        goto LABEL_14;
      goto LABEL_20;
    }
    if ( v13 <= 0xD3u )
      goto LABEL_23;
  }
  else
  {
    v18 = sub_3007030((__int64)&v19);
    if ( sub_30070B0((__int64)&v19) )
      goto LABEL_23;
    if ( !v18 )
      goto LABEL_12;
  }
  v15 = *(_QWORD *)(v8 + 96);
  v16 = *(_DWORD *)(v15 + 32);
  if ( !a1[16] )
  {
LABEL_14:
    v17 = *(_QWORD **)(v15 + 24);
    if ( v16 > 0x40 )
      return (*v17 & 1) == 0;
    else
      return ((unsigned __int8)v17 & 1) == 0;
  }
LABEL_20:
  if ( v16 <= 0x40 )
    return *(_QWORD *)(v15 + 24) == 0;
  else
    return v16 == (unsigned int)sub_C444A0(v15 + 24);
}
