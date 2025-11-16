// Function: sub_BA2D00
// Address: 0xba2d00
//
__int64 __fastcall sub_BA2D00(__int64 a1, __int64 a2)
{
  unsigned __int16 v3; // ax
  __int64 v4; // rdx
  unsigned __int8 v5; // al
  unsigned __int8 v6; // al
  __int64 v7; // rcx
  unsigned __int8 v8; // al
  __int64 v9; // rdx
  int v10; // r14d
  __int64 v11; // r13
  int v12; // r14d
  unsigned int i; // ebx
  __int64 *v14; // r15
  __int64 v15; // rcx
  unsigned int v16; // ebx
  __int64 v17; // rax
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  __int64 result; // rax
  unsigned int v21; // esi
  int v22; // eax
  _QWORD *v23; // rdx
  int v24; // eax
  __int64 v25; // [rsp+0h] [rbp-90h]
  int v26; // [rsp+8h] [rbp-88h]
  int v27; // [rsp+14h] [rbp-7Ch]
  __int64 v28[2]; // [rsp+18h] [rbp-78h] BYREF
  _QWORD *v29; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v30; // [rsp+30h] [rbp-60h] BYREF
  __int64 v31; // [rsp+38h] [rbp-58h] BYREF
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  __int8 v33[8]; // [rsp+48h] [rbp-48h] BYREF
  __int64 v34[8]; // [rsp+50h] [rbp-40h] BYREF

  v28[0] = a1;
  v3 = sub_AF18C0(a1);
  v4 = a1 - 16;
  LODWORD(v30) = v3;
  v5 = *(_BYTE *)(a1 - 16);
  if ( (v5 & 2) != 0 )
  {
    v31 = **(_QWORD **)(a1 - 32);
    v6 = *(_BYTE *)(a1 - 16);
    if ( (v6 & 2) != 0 )
    {
LABEL_3:
      v7 = *(_QWORD *)(a1 - 32);
      goto LABEL_4;
    }
  }
  else
  {
    v31 = *(_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF));
    v6 = *(_BYTE *)(a1 - 16);
    if ( (v6 & 2) != 0 )
      goto LABEL_3;
  }
  v7 = v4 - 8LL * ((v6 >> 2) & 0xF);
LABEL_4:
  v32 = *(_QWORD *)(v7 + 8);
  v33[0] = *(_BYTE *)(a1 + 1) >> 7;
  v8 = *(_BYTE *)(a1 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(_QWORD *)(a1 - 32);
  else
    v9 = v4 - 8LL * ((v8 >> 2) & 0xF);
  v10 = *(_DWORD *)(a2 + 24);
  v11 = *(_QWORD *)(a2 + 8);
  v34[0] = *(_QWORD *)(v9 + 16);
  if ( !v10 )
  {
LABEL_20:
    if ( (unsigned __int8)sub_AFEAC0(a2, v28, &v29) )
      return v28[0];
    v21 = *(_DWORD *)(a2 + 24);
    v22 = *(_DWORD *)(a2 + 16);
    v23 = v29;
    ++*(_QWORD *)a2;
    v24 = v22 + 1;
    v30 = v23;
    if ( 4 * v24 >= 3 * v21 )
    {
      v21 *= 2;
    }
    else if ( v21 - *(_DWORD *)(a2 + 20) - v24 > v21 >> 3 )
    {
LABEL_28:
      *(_DWORD *)(a2 + 16) = v24;
      if ( *v23 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v23 = v28[0];
      return v28[0];
    }
    sub_B0ADF0(a2, v21);
    sub_AFEAC0(a2, v28, &v30);
    v23 = v30;
    v24 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_28;
  }
  v12 = v10 - 1;
  v27 = 1;
  for ( i = v12 & sub_AF9230((int *)&v30, &v31, &v32, v33, v34); ; i = v12 & v16 )
  {
    v14 = (__int64 *)(v11 + 8LL * i);
    v15 = *v14;
    if ( *v14 == -8192 )
      goto LABEL_12;
    if ( v15 == -4096 )
      goto LABEL_20;
    v25 = *v14;
    v26 = (int)v30;
    if ( v26 == (unsigned __int16)sub_AF18C0(*v14) )
    {
      v17 = sub_AF5140(v25, 0);
      if ( v31 == v17 )
      {
        v18 = sub_A17150((_BYTE *)(v25 - 16));
        if ( v32 == *((_QWORD *)v18 + 1) && v33[0] == *(_BYTE *)(v25 + 1) >> 7 )
        {
          v19 = sub_A17150((_BYTE *)(v25 - 16));
          if ( v34[0] == *((_QWORD *)v19 + 2) )
            break;
        }
      }
    }
    v15 = *v14;
LABEL_12:
    if ( v15 == -4096 )
      goto LABEL_20;
    v16 = v27 + i;
    ++v27;
  }
  if ( v14 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
    goto LABEL_20;
  result = *v14;
  if ( !*v14 )
    goto LABEL_20;
  return result;
}
