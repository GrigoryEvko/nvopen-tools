// Function: sub_BA2170
// Address: 0xba2170
//
__int64 __fastcall sub_BA2170(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned __int8 v4; // dl
  __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned __int8 v7; // dl
  __int64 *v8; // rcx
  int v9; // r13d
  __int64 v10; // r14
  int v11; // eax
  int v12; // ecx
  int v13; // esi
  unsigned int i; // r12d
  __int64 *v15; // r15
  __int64 v16; // r13
  _BYTE *v17; // rax
  unsigned int v18; // r12d
  __int64 v20; // r8
  __int64 v21; // rax
  _BYTE *v22; // rax
  unsigned int v23; // esi
  int v24; // eax
  _QWORD *v25; // rdx
  int v26; // eax
  __int64 v27; // [rsp+0h] [rbp-80h]
  int v28; // [rsp+Ch] [rbp-74h]
  __int64 v29[2]; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v30; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v31; // [rsp+30h] [rbp-50h] BYREF
  __int64 v32; // [rsp+38h] [rbp-48h] BYREF
  int v33; // [rsp+40h] [rbp-40h] BYREF
  int v34[15]; // [rsp+44h] [rbp-3Ch] BYREF

  v2 = a1 - 16;
  v29[0] = a1;
  v4 = *(_BYTE *)(a1 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(a1 - 32);
  else
    v5 = v2 - 8LL * ((v4 >> 2) & 0xF);
  v31 = *(_QWORD **)(v5 + 8);
  v6 = a1;
  if ( *(_BYTE *)a1 != 16 )
  {
    v7 = *(_BYTE *)(a1 - 16);
    if ( (v7 & 2) != 0 )
      v8 = *(__int64 **)(a1 - 32);
    else
      v8 = (__int64 *)(v2 - 8LL * ((v7 >> 2) & 0xF));
    v6 = *v8;
  }
  v9 = *(_DWORD *)(a2 + 24);
  v10 = *(_QWORD *)(a2 + 8);
  v32 = v6;
  v33 = *(_DWORD *)(a1 + 4);
  v34[0] = *(unsigned __int16 *)(a1 + 16);
  if ( v9 )
  {
    v11 = sub_AF7510((__int64 *)&v31, &v32, &v33, v34);
    v12 = v9 - 1;
    v13 = 1;
    for ( i = (v9 - 1) & v11; ; i = v12 & v18 )
    {
      v15 = (__int64 *)(v10 + 8LL * i);
      v16 = *v15;
      if ( *v15 == -4096 )
        break;
      if ( v16 != -8192 )
      {
        v28 = v12;
        v17 = sub_A17150((_BYTE *)(v16 - 16));
        v12 = v28;
        if ( v31 == *((_QWORD **)v17 + 1) )
        {
          v20 = v32;
          v21 = v16;
          if ( *(_BYTE *)v16 != 16 )
          {
            v27 = v32;
            v22 = sub_A17150((_BYTE *)(v16 - 16));
            v20 = v27;
            v21 = *(_QWORD *)v22;
            v12 = v28;
          }
          if ( v20 == v21 && v33 == *(_DWORD *)(v16 + 4) && v34[0] == *(unsigned __int16 *)(v16 + 16) )
          {
            if ( v15 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
              break;
            return v16;
          }
        }
      }
      v18 = v13 + i;
      ++v13;
    }
  }
  if ( !(unsigned __int8)sub_AFE170(a2, v29, &v30) )
  {
    v23 = *(_DWORD *)(a2 + 24);
    v24 = *(_DWORD *)(a2 + 16);
    v25 = v30;
    ++*(_QWORD *)a2;
    v26 = v24 + 1;
    v31 = v25;
    if ( 4 * v26 >= 3 * v23 )
    {
      v23 *= 2;
    }
    else if ( v23 - *(_DWORD *)(a2 + 20) - v26 > v23 >> 3 )
    {
LABEL_26:
      *(_DWORD *)(a2 + 16) = v26;
      if ( *v25 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v25 = v29[0];
      return v29[0];
    }
    sub_B084F0(a2, v23);
    sub_AFE170(a2, v29, &v31);
    v25 = v31;
    v26 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_26;
  }
  return v29[0];
}
