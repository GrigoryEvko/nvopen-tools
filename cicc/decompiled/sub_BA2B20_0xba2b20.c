// Function: sub_BA2B20
// Address: 0xba2b20
//
__int64 __fastcall sub_BA2B20(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  int v5; // r13d
  __int64 v6; // r14
  int v7; // eax
  int v8; // ecx
  int v9; // r8d
  unsigned int i; // ebx
  __int64 *v11; // r15
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned int v14; // ebx
  _BYTE *v16; // rax
  unsigned int v17; // esi
  int v18; // eax
  _QWORD *v19; // rdx
  int v20; // eax
  int v21; // [rsp+0h] [rbp-70h]
  int v22; // [rsp+4h] [rbp-6Ch]
  __int64 v23[2]; // [rsp+8h] [rbp-68h] BYREF
  _QWORD *v24; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h] BYREF
  __int8 v27[64]; // [rsp+30h] [rbp-40h] BYREF

  v23[0] = a1;
  v25 = (_QWORD *)sub_AF5140(a1, 0);
  v3 = *(_BYTE *)(a1 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_QWORD *)(a1 - 32);
  else
    v4 = a1 - 16 - 8LL * ((v3 >> 2) & 0xF);
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v26 = *(_QWORD *)(v4 + 8);
  v27[0] = *(_BYTE *)(a1 + 1) >> 7;
  if ( v5 )
  {
    v7 = sub_AFA150((__int64 *)&v25, &v26, v27);
    v8 = v5 - 1;
    v9 = 1;
    for ( i = (v5 - 1) & v7; ; i = v8 & v14 )
    {
      v11 = (__int64 *)(v6 + 8LL * i);
      v12 = *v11;
      if ( *v11 == -4096 )
        break;
      if ( v12 != -8192 )
      {
        v21 = v9;
        v22 = v8;
        v13 = sub_AF5140(*v11, 0);
        v8 = v22;
        v9 = v21;
        if ( v25 == (_QWORD *)v13 )
        {
          v16 = sub_A17150((_BYTE *)(v12 - 16));
          v8 = v22;
          v9 = v21;
          if ( v26 == *((_QWORD *)v16 + 1) && v27[0] == *(_BYTE *)(v12 + 1) >> 7 )
          {
            if ( v11 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
              break;
            return v12;
          }
        }
      }
      v14 = v9 + i;
      ++v9;
    }
  }
  if ( !(unsigned __int8)sub_AFE960(a2, v23, &v24) )
  {
    v17 = *(_DWORD *)(a2 + 24);
    v18 = *(_DWORD *)(a2 + 16);
    v19 = v24;
    ++*(_QWORD *)a2;
    v20 = v18 + 1;
    v25 = v19;
    if ( 4 * v20 >= 3 * v17 )
    {
      v17 *= 2;
    }
    else if ( v17 - *(_DWORD *)(a2 + 20) - v20 > v17 >> 3 )
    {
LABEL_18:
      *(_DWORD *)(a2 + 16) = v20;
      if ( *v19 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v19 = v23[0];
      return v23[0];
    }
    sub_B0A7B0(a2, v17);
    sub_AFE960(a2, v23, &v25);
    v19 = v25;
    v20 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_18;
  }
  return v23[0];
}
