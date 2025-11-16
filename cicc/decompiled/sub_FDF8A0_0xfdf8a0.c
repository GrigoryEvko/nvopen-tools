// Function: sub_FDF8A0
// Address: 0xfdf8a0
//
char __fastcall sub_FDF8A0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  char v7; // cl
  __int64 v8; // rdi
  int v9; // edx
  unsigned int v10; // r9d
  int *v11; // rax
  int v12; // r10d
  __int64 v13; // rdx
  unsigned int *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rax
  _DWORD *v18; // rdi
  __int64 v19; // r14
  __int64 i; // r15
  unsigned int v21; // r15d
  unsigned int v22; // esi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // eax
  int v27; // r11d
  int j; // [rsp+14h] [rbp-6Ch]
  __int64 v30; // [rsp+18h] [rbp-68h]
  int v31; // [rsp+24h] [rbp-5Ch] BYREF
  __int64 v32; // [rsp+28h] [rbp-58h] BYREF
  __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-48h]
  int v35; // [rsp+48h] [rbp-38h]

  v7 = *(_BYTE *)(a1 + 56) & 1;
  if ( v7 )
  {
    v8 = a1 + 64;
    v9 = 3;
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 72);
    v8 = *(_QWORD *)(a1 + 64);
    if ( !(_DWORD)v24 )
      goto LABEL_21;
    v9 = v24 - 1;
  }
  v10 = v9 & (37 * *a2);
  v11 = (int *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( *v11 == *a2 )
    goto LABEL_4;
  v26 = 1;
  while ( v12 != -1 )
  {
    v27 = v26 + 1;
    v10 = v9 & (v26 + v10);
    v11 = (int *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( *a2 == *v11 )
      goto LABEL_4;
    v26 = v27;
  }
  if ( v7 )
  {
    v25 = 64;
    goto LABEL_22;
  }
  v24 = *(unsigned int *)(a1 + 72);
LABEL_21:
  v25 = 16 * v24;
LABEL_22:
  v11 = (int *)(v8 + v25);
LABEL_4:
  v13 = 64;
  if ( !v7 )
    v13 = 16LL * *(unsigned int *)(a1 + 72);
  if ( v11 != (int *)(v8 + v13) )
  {
    v14 = (unsigned int *)*((_QWORD *)v11 + 1);
    v15 = *(_QWORD *)(*(_QWORD *)a1 + 64LL) + 24LL * (unsigned int)*a2;
    v16 = *(_QWORD *)(v15 + 8);
    if ( v16 )
    {
      v17 = *(unsigned int *)(v16 + 12);
      v18 = *(_DWORD **)(v16 + 96);
      if ( (unsigned int)v17 > 1 )
      {
        LOBYTE(v11) = sub_FDC990(v18, &v18[v17], (_DWORD *)v15);
        if ( (_BYTE)v11 )
        {
LABEL_10:
          if ( *(_BYTE *)(v16 + 8) )
          {
            v19 = *(_QWORD *)(v16 + 16);
            for ( i = v19 + 16LL * *(unsigned int *)(v16 + 24); i != v19; v19 += 16 )
              LOBYTE(v11) = sub_FEB360(a1, v14, v19, a3);
            return (char)v11;
          }
        }
      }
      else
      {
        LODWORD(v11) = *v18;
        if ( *(_DWORD *)v15 == *v18 )
          goto LABEL_10;
      }
    }
    v32 = *(_QWORD *)(*(_QWORD *)(a4 + 136) + 8LL * *v14);
    sub_FDC9F0((__int64)&v33, &v32);
    v21 = v34;
    v30 = v33;
    LOBYTE(v11) = v35;
    for ( j = v35; j != v21; LOBYTE(v11) = sub_FEB360(a1, v14, &v31, a3) )
    {
      v22 = v21++;
      v23 = sub_B46EC0(v30, v22);
      v31 = sub_FDD0F0(a4, v23);
    }
  }
  return (char)v11;
}
