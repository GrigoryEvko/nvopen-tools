// Function: sub_BA23B0
// Address: 0xba23b0
//
__int64 __fastcall sub_BA23B0(__int64 a1, __int64 a2)
{
  unsigned __int8 v4; // al
  _BYTE *v5; // rdi
  _BYTE *v6; // rdx
  __int64 v7; // rax
  int v8; // r13d
  __int64 v9; // r14
  int v10; // eax
  int v11; // ecx
  int v12; // esi
  unsigned int i; // ebx
  __int64 *v14; // r15
  __int64 v15; // r13
  _BYTE *v16; // rax
  unsigned int v17; // ebx
  __int64 v19; // r8
  __int64 v20; // rax
  _BYTE *v21; // rax
  unsigned int v22; // esi
  int v23; // eax
  _QWORD *v24; // rdx
  int v25; // eax
  __int64 v26; // [rsp+0h] [rbp-80h]
  int v27; // [rsp+Ch] [rbp-74h]
  __int64 v28[2]; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v29; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h] BYREF
  int v32[16]; // [rsp+40h] [rbp-40h] BYREF

  v28[0] = a1;
  v4 = *(_BYTE *)(a1 - 16);
  v5 = (_BYTE *)(a1 - 16);
  if ( (v4 & 2) != 0 )
    v6 = *(_BYTE **)(a1 - 32);
  else
    v6 = &v5[-8 * ((v4 >> 2) & 0xF)];
  v30 = (_QWORD *)*((_QWORD *)v6 + 1);
  v7 = a1;
  if ( *(_BYTE *)a1 != 16 )
    v7 = *(_QWORD *)sub_A17150(v5);
  v8 = *(_DWORD *)(a2 + 24);
  v9 = *(_QWORD *)(a2 + 8);
  v31 = v7;
  v32[0] = *(_DWORD *)(a1 + 4);
  if ( v8 )
  {
    v10 = sub_AF7750((__int64 *)&v30, &v31, v32);
    v11 = v8 - 1;
    v12 = 1;
    for ( i = (v8 - 1) & v10; ; i = v11 & v17 )
    {
      v14 = (__int64 *)(v9 + 8LL * i);
      v15 = *v14;
      if ( *v14 == -4096 )
        break;
      if ( v15 != -8192 )
      {
        v27 = v11;
        v16 = sub_A17150((_BYTE *)(v15 - 16));
        v11 = v27;
        if ( v30 == *((_QWORD **)v16 + 1) )
        {
          v19 = v31;
          v20 = v15;
          if ( *(_BYTE *)v15 != 16 )
          {
            v26 = v31;
            v21 = sub_A17150((_BYTE *)(v15 - 16));
            v19 = v26;
            v20 = *(_QWORD *)v21;
            v11 = v27;
          }
          if ( v19 == v20 && v32[0] == *(_DWORD *)(v15 + 4) )
          {
            if ( v14 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
              break;
            return v15;
          }
        }
      }
      v17 = v12 + i;
      ++v12;
    }
  }
  if ( !(unsigned __int8)sub_AFE2D0(a2, v28, &v29) )
  {
    v22 = *(_DWORD *)(a2 + 24);
    v23 = *(_DWORD *)(a2 + 16);
    v24 = v29;
    ++*(_QWORD *)a2;
    v25 = v23 + 1;
    v30 = v24;
    if ( 4 * v25 >= 3 * v22 )
    {
      v22 *= 2;
    }
    else if ( v22 - *(_DWORD *)(a2 + 20) - v25 > v22 >> 3 )
    {
LABEL_22:
      *(_DWORD *)(a2 + 16) = v25;
      if ( *v24 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v24 = v28[0];
      return v28[0];
    }
    sub_B08BF0(a2, v22);
    sub_AFE2D0(a2, v28, &v30);
    v24 = v30;
    v25 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_22;
  }
  return v28[0];
}
