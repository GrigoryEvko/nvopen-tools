// Function: sub_BA01A0
// Address: 0xba01a0
//
__int64 __fastcall sub_BA01A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned __int8 v4; // dl
  unsigned __int8 v5; // dl
  __int64 v6; // rax
  int v7; // r15d
  __int64 v8; // r14
  int v9; // r15d
  int v10; // eax
  int v11; // ecx
  unsigned int i; // r13d
  __int64 v13; // r12
  _BYTE *v14; // rax
  unsigned int v15; // r13d
  _BYTE *v16; // rax
  __int64 result; // rax
  unsigned int v18; // esi
  int v19; // eax
  _QWORD *v20; // rdx
  int v21; // eax
  int v22; // [rsp+4h] [rbp-6Ch]
  __int64 v23[2]; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v24; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v25; // [rsp+30h] [rbp-40h] BYREF
  __int64 v26[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = a1 - 16;
  v23[0] = a1;
  v4 = *(_BYTE *)(a1 - 16);
  if ( (v4 & 2) != 0 )
  {
    v25 = **(_QWORD ***)(a1 - 32);
    v5 = *(_BYTE *)(a1 - 16);
    if ( (v5 & 2) != 0 )
    {
LABEL_3:
      v6 = *(_QWORD *)(a1 - 32);
      goto LABEL_4;
    }
  }
  else
  {
    v25 = *(_QWORD **)(v2 - 8LL * ((v4 >> 2) & 0xF));
    v5 = *(_BYTE *)(a1 - 16);
    if ( (v5 & 2) != 0 )
      goto LABEL_3;
  }
  v6 = v2 - 8LL * ((v5 >> 2) & 0xF);
LABEL_4:
  v7 = *(_DWORD *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 8);
  v26[0] = *(_QWORD *)(v6 + 8);
  if ( v7 )
  {
    v9 = v7 - 1;
    v10 = sub_AF7B60((__int64 *)&v25, v26);
    v11 = 1;
    for ( i = v9 & v10; ; i = v9 & v15 )
    {
      v13 = *(_QWORD *)(v8 + 8LL * i);
      if ( v13 == -4096 )
        break;
      if ( v13 != -8192 )
      {
        v22 = v11;
        v14 = sub_A17150((_BYTE *)(v13 - 16));
        v11 = v22;
        if ( v25 == *(_QWORD **)v14 )
        {
          v16 = sub_A17150((_BYTE *)(v13 - 16));
          v11 = v22;
          if ( v26[0] == *((_QWORD *)v16 + 1) )
          {
            if ( v8 + 8LL * i != *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24) )
            {
              result = v13;
              if ( v13 )
                return result;
            }
            break;
          }
        }
      }
      v15 = v11 + i;
      ++v11;
    }
  }
  if ( !(unsigned __int8)sub_AFF3C0(a2, v23, &v24) )
  {
    v18 = *(_DWORD *)(a2 + 24);
    v19 = *(_DWORD *)(a2 + 16);
    v20 = v24;
    ++*(_QWORD *)a2;
    v21 = v19 + 1;
    v25 = v20;
    if ( 4 * v21 >= 3 * v18 )
    {
      v18 *= 2;
    }
    else if ( v18 - *(_DWORD *)(a2 + 20) - v21 > v18 >> 3 )
    {
LABEL_20:
      *(_DWORD *)(a2 + 16) = v21;
      if ( *v20 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v20 = v23[0];
      return v23[0];
    }
    sub_B0EB00(a2, v18);
    sub_AFF3C0(a2, v23, &v25);
    v20 = v25;
    v21 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_20;
  }
  return v23[0];
}
