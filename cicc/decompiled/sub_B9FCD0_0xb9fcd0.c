// Function: sub_B9FCD0
// Address: 0xb9fcd0
//
__int64 __fastcall sub_B9FCD0(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rdi
  unsigned __int8 v6; // al
  int v7; // eax
  __int64 v8; // rdx
  int v9; // r14d
  __int64 v10; // r15
  int v11; // r14d
  int v12; // esi
  unsigned int i; // r13d
  __int64 v14; // rbx
  unsigned int v15; // r13d
  unsigned int v17; // esi
  int v18; // eax
  _QWORD *v19; // rdx
  int v20; // eax
  _BYTE *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  int v24; // eax
  __int64 v25; // rdx
  _BYTE *v26; // rax
  __int64 v27; // [rsp+0h] [rbp-80h]
  __int64 v28[2]; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v29; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h] BYREF
  __int64 v32; // [rsp+40h] [rbp-40h] BYREF
  __int8 v33[56]; // [rsp+48h] [rbp-38h] BYREF

  v28[0] = a1;
  v4 = *(_DWORD *)(a1 + 4);
  v5 = a1 - 16;
  LODWORD(v30) = v4;
  HIDWORD(v30) = *(unsigned __int16 *)(v5 + 18);
  v6 = *(_BYTE *)(a1 - 16);
  if ( (v6 & 2) != 0 )
  {
    v31 = **(_QWORD **)(a1 - 32);
    if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    {
LABEL_3:
      v7 = *(_DWORD *)(a1 - 24);
      goto LABEL_4;
    }
  }
  else
  {
    v31 = *(_QWORD *)(v5 - 8LL * ((v6 >> 2) & 0xF));
    if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
      goto LABEL_3;
  }
  v7 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
LABEL_4:
  v8 = 0;
  if ( v7 == 2 )
    v8 = *((_QWORD *)sub_A17150((_BYTE *)v5) + 1);
  v9 = *(_DWORD *)(a2 + 24);
  v10 = *(_QWORD *)(a2 + 8);
  v32 = v8;
  v33[0] = *(_BYTE *)(a1 + 1) >> 7;
  if ( v9 )
  {
    v11 = v9 - 1;
    v12 = 1;
    for ( i = v11 & sub_AF71E0((int *)&v30, (int *)&v30 + 1, &v31, &v32, v33); ; i = v11 & v15 )
    {
      v14 = *(_QWORD *)(v10 + 8LL * i);
      if ( v14 == -4096 )
        break;
      if ( v14 != -8192 && v30 == (_QWORD *)__PAIR64__(*(unsigned __int16 *)(v14 + 2), *(_DWORD *)(v14 + 4)) )
      {
        v21 = sub_A17150((_BYTE *)(v14 - 16));
        v22 = v10 + 8LL * i;
        if ( v31 == *(_QWORD *)v21 )
        {
          v23 = v32;
          if ( (*(_BYTE *)(v14 - 16) & 2) != 0 )
            v24 = *(_DWORD *)(v14 - 24);
          else
            v24 = (*(_WORD *)(v14 - 16) >> 6) & 0xF;
          v25 = 0;
          if ( v24 == 2 )
          {
            v27 = v32;
            v26 = sub_A17150((_BYTE *)(v14 - 16));
            v23 = v27;
            v25 = *((_QWORD *)v26 + 1);
            v22 = v10 + 8LL * i;
          }
          if ( v23 == v25 && v33[0] == (unsigned __int8)BYTE1(*(_QWORD *)v14) >> 7 )
          {
            if ( v22 == *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24) )
              break;
            return v14;
          }
        }
      }
      v15 = v12 + i;
      ++v12;
    }
  }
  if ( !(unsigned __int8)sub_AFC140(a2, v28, &v29) )
  {
    v17 = *(_DWORD *)(a2 + 24);
    v18 = *(_DWORD *)(a2 + 16);
    v19 = v29;
    ++*(_QWORD *)a2;
    v20 = v18 + 1;
    v30 = v19;
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
      *v19 = v28[0];
      return v28[0];
    }
    sub_B01330(a2, v17);
    sub_AFC140(a2, v28, &v30);
    v19 = v30;
    v20 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_18;
  }
  return v28[0];
}
