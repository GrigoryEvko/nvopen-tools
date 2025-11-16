// Function: sub_BA0DD0
// Address: 0xba0dd0
//
__int64 __fastcall sub_BA0DD0(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  int v5; // eax
  int v6; // r14d
  __int64 v7; // r13
  int v8; // r14d
  unsigned int i; // ebx
  __int64 *v10; // r15
  __int64 v11; // rdx
  unsigned int v12; // ebx
  __int64 v13; // rax
  __int64 result; // rax
  unsigned int v15; // esi
  int v16; // eax
  _QWORD *v17; // rdx
  int v18; // eax
  __int64 v19; // [rsp+0h] [rbp-90h]
  int v20; // [rsp+8h] [rbp-88h]
  int v21; // [rsp+8h] [rbp-88h]
  int v22; // [rsp+14h] [rbp-7Ch]
  __int64 v23[2]; // [rsp+18h] [rbp-78h] BYREF
  _QWORD *v24; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v25; // [rsp+30h] [rbp-60h] BYREF
  __int64 v26; // [rsp+38h] [rbp-58h] BYREF
  __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  int v28; // [rsp+48h] [rbp-48h] BYREF
  int v29; // [rsp+4Ch] [rbp-44h] BYREF
  int v30; // [rsp+50h] [rbp-40h]
  int v31; // [rsp+54h] [rbp-3Ch]

  v23[0] = a1;
  LODWORD(v25) = (unsigned __int16)sub_AF18C0(a1);
  v3 = *(_BYTE *)(a1 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_QWORD *)(a1 - 32);
  else
    v4 = a1 - 16 - 8LL * ((v3 >> 2) & 0xF);
  v26 = *(_QWORD *)(v4 + 16);
  v27 = *(_QWORD *)(a1 + 24);
  v5 = sub_AF18D0(a1);
  v6 = *(_DWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 8);
  v28 = v5;
  v29 = *(_DWORD *)(a1 + 44);
  v30 = *(_DWORD *)(a1 + 40);
  v31 = *(_DWORD *)(a1 + 20);
  if ( !v6 )
  {
LABEL_19:
    if ( (unsigned __int8)sub_AFCEE0(a2, v23, &v24) )
      return v23[0];
    v15 = *(_DWORD *)(a2 + 24);
    v16 = *(_DWORD *)(a2 + 16);
    v17 = v24;
    ++*(_QWORD *)a2;
    v18 = v16 + 1;
    v25 = v17;
    if ( 4 * v18 >= 3 * v15 )
    {
      v15 *= 2;
    }
    else if ( v15 - *(_DWORD *)(a2 + 20) - v18 > v15 >> 3 )
    {
LABEL_25:
      *(_DWORD *)(a2 + 16) = v18;
      if ( *v17 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v17 = v23[0];
      return v23[0];
    }
    sub_B049F0(a2, v15);
    sub_AFCEE0(a2, v23, &v25);
    v17 = v25;
    v18 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_25;
  }
  v8 = v6 - 1;
  v22 = 1;
  for ( i = v8 & sub_AF9B00((int *)&v25, &v26, &v27, &v28, &v29); ; i = v8 & v12 )
  {
    v10 = (__int64 *)(v7 + 8LL * i);
    v11 = *v10;
    if ( *v10 == -8192 )
      goto LABEL_9;
    if ( v11 == -4096 )
      goto LABEL_19;
    v19 = *v10;
    v20 = (int)v25;
    if ( v20 == (unsigned __int16)sub_AF18C0(*v10) )
    {
      v13 = sub_AF5140(v19, 2u);
      if ( v26 == v13 && v27 == *(_QWORD *)(v19 + 24) )
      {
        v21 = v28;
        if ( v21 == (unsigned int)sub_AF18D0(v19)
          && v29 == *(_DWORD *)(v19 + 44)
          && v30 == *(_DWORD *)(v19 + 40)
          && v31 == *(_DWORD *)(v19 + 20) )
        {
          break;
        }
      }
    }
    v11 = *v10;
LABEL_9:
    if ( v11 == -4096 )
      goto LABEL_19;
    v12 = v22 + i;
    ++v22;
  }
  if ( v10 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
    goto LABEL_19;
  result = *v10;
  if ( !*v10 )
    goto LABEL_19;
  return result;
}
