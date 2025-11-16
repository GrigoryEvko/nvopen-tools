// Function: sub_BA1960
// Address: 0xba1960
//
__int64 __fastcall sub_BA1960(__int64 a1, __int64 a2)
{
  int v3; // edx
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // r14d
  int v8; // r14d
  int v9; // r15d
  int v10; // eax
  __int64 v11; // rcx
  unsigned int i; // r12d
  __int64 v13; // r13
  unsigned int v14; // r12d
  _BYTE *v16; // rax
  unsigned int v17; // esi
  int v18; // eax
  _QWORD *v19; // rdx
  int v20; // eax
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+10h] [rbp-60h]
  __int64 v23; // [rsp+10h] [rbp-60h]
  __int64 v24[2]; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v25; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v26; // [rsp+30h] [rbp-40h] BYREF
  __int64 v27[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a1 + 20);
  v24[0] = a1;
  LODWORD(v26) = v3;
  BYTE4(v26) = *(_BYTE *)(a1 + 44);
  v4 = *(_BYTE *)(a1 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(a1 - 32);
  else
    v5 = a1 - 16 - 8LL * ((v4 >> 2) & 0xF);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_DWORD *)(a2 + 24);
  v27[0] = *(_QWORD *)(v5 + 24);
  v22 = v6;
  if ( v7 )
  {
    v8 = v7 - 1;
    v9 = 1;
    v10 = sub_AF8410((int *)&v26, (__int8 *)&v26 + 4, v27);
    v11 = v22;
    for ( i = v8 & v10; ; i = v8 & v14 )
    {
      v13 = *(_QWORD *)(v11 + 8LL * i);
      if ( v13 == -4096 )
        break;
      if ( v13 != -8192 && (_DWORD)v26 == *(_DWORD *)(v13 + 20) && BYTE4(v26) == *(_BYTE *)(v13 + 44) )
      {
        v21 = v11 + 8LL * i;
        v23 = v11;
        v16 = sub_A17150((_BYTE *)(v13 - 16));
        v11 = v23;
        if ( v27[0] == *((_QWORD *)v16 + 3) )
        {
          if ( v21 == *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24) )
            break;
          return v13;
        }
      }
      v14 = v9 + i;
      ++v9;
    }
  }
  if ( !(unsigned __int8)sub_AFD910(a2, v24, &v25) )
  {
    v17 = *(_DWORD *)(a2 + 24);
    v18 = *(_DWORD *)(a2 + 16);
    v19 = v25;
    ++*(_QWORD *)a2;
    v20 = v18 + 1;
    v26 = v19;
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
      *v19 = v24[0];
      return v24[0];
    }
    sub_B06E50(a2, v17);
    sub_AFD910(a2, v24, &v26);
    v19 = v26;
    v20 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_18;
  }
  return v24[0];
}
