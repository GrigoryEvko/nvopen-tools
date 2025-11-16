// Function: sub_2C16520
// Address: 0x2c16520
//
__int64 __fastcall sub_2C16520(__int64 a1)
{
  __int64 v1; // rbx
  __int64 *v2; // rax
  __int64 v3; // r14
  __int64 v4; // r9
  __int64 v5; // r15
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v14; // [rsp+10h] [rbp-80h]
  char v15; // [rsp+1Fh] [rbp-71h]
  __int64 v16; // [rsp+20h] [rbp-70h]
  __int64 v17; // [rsp+28h] [rbp-68h]
  __int64 v18; // [rsp+30h] [rbp-60h] BYREF
  __int64 v19; // [rsp+38h] [rbp-58h] BYREF
  __int64 v20; // [rsp+40h] [rbp-50h] BYREF
  __int64 v21; // [rsp+48h] [rbp-48h] BYREF
  __int64 v22; // [rsp+50h] [rbp-40h] BYREF
  __int64 v23; // [rsp+58h] [rbp-38h]

  v1 = 0;
  v16 = *(_QWORD *)(a1 + 136);
  v2 = *(__int64 **)(a1 + 48);
  v3 = *v2;
  v17 = v2[1];
  if ( *(_BYTE *)(a1 + 161) )
    v1 = v2[*(_DWORD *)(a1 + 56) - 1];
  v18 = *(_QWORD *)(a1 + 88);
  if ( v18 )
    sub_2AAAFA0(&v18);
  v5 = sub_22077B0(0xA8u);
  if ( v5 )
  {
    v15 = *(_BYTE *)(a1 + 160);
    v14 = *(_QWORD *)(a1 + 152);
    v19 = v18;
    if ( v18 )
    {
      sub_2AAAFA0(&v19);
      v22 = v3;
      v23 = v17;
      v20 = v19;
      if ( v19 )
      {
        sub_2AAAFA0(&v20);
        v21 = v20;
        if ( v20 )
          sub_2AAAFA0(&v21);
        goto LABEL_10;
      }
    }
    else
    {
      v22 = v3;
      v20 = 0;
      v23 = v17;
    }
    v21 = 0;
LABEL_10:
    sub_2AAF310(v5, 7, &v22, 2, &v21, v4);
    sub_9C6650(&v21);
    sub_2BF0340(v5 + 96, 1, v16, v5, v6, v7);
    *(_QWORD *)v5 = &unk_4A231C8;
    *(_QWORD *)(v5 + 40) = &unk_4A23200;
    *(_QWORD *)(v5 + 96) = &unk_4A23238;
    sub_9C6650(&v20);
    *(_QWORD *)v5 = &unk_4A24438;
    *(_QWORD *)(v5 + 96) = &unk_4A244A8;
    *(_QWORD *)(v5 + 40) = &unk_4A24470;
    *(_QWORD *)(v5 + 152) = v14;
    *(_BYTE *)(v5 + 160) = v15;
    if ( v1 )
    {
      v10 = *(unsigned int *)(v5 + 56);
      v11 = *(unsigned int *)(v5 + 60);
      *(_BYTE *)(v5 + 161) = 1;
      if ( v10 + 1 > v11 )
      {
        sub_C8D5F0(v5 + 48, (const void *)(v5 + 64), v10 + 1, 8u, v8, v9);
        v10 = *(unsigned int *)(v5 + 56);
      }
      *(_QWORD *)(*(_QWORD *)(v5 + 48) + 8 * v10) = v1;
      ++*(_DWORD *)(v5 + 56);
      v12 = *(unsigned int *)(v1 + 24);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v1 + 28) )
      {
        sub_C8D5F0(v1 + 16, (const void *)(v1 + 32), v12 + 1, 8u, v8, v9);
        v12 = *(unsigned int *)(v1 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(v1 + 16) + 8 * v12) = v5 + 40;
      ++*(_DWORD *)(v1 + 24);
    }
    else
    {
      *(_BYTE *)(v5 + 161) = 0;
    }
    sub_9C6650(&v19);
  }
  sub_9C6650(&v18);
  return v5;
}
