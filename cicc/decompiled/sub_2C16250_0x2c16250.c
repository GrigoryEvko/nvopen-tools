// Function: sub_2C16250
// Address: 0x2c16250
//
__int64 __fastcall sub_2C16250(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r15
  __int64 v3; // rbx
  __int64 v4; // r9
  __int64 v5; // r14
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  char v14; // [rsp+1Fh] [rbp-71h]
  __int64 v15; // [rsp+20h] [rbp-70h]
  __int64 v16; // [rsp+28h] [rbp-68h]
  __int64 v17; // [rsp+30h] [rbp-60h] BYREF
  __int64 v18; // [rsp+38h] [rbp-58h] BYREF
  __int64 v19; // [rsp+40h] [rbp-50h] BYREF
  __int64 v20; // [rsp+48h] [rbp-48h] BYREF
  __int64 v21; // [rsp+50h] [rbp-40h] BYREF
  __int64 v22[7]; // [rsp+58h] [rbp-38h] BYREF

  v15 = *(_QWORD *)(a1 + 136);
  v1 = *(__int64 **)(a1 + 48);
  v2 = *v1;
  v3 = v1[1];
  v16 = *(_QWORD *)(a1 + 152);
  v17 = *(_QWORD *)(a1 + 88);
  if ( v17 )
    sub_2AAAFA0(&v17);
  v5 = sub_22077B0(0xA8u);
  if ( v5 )
  {
    v14 = *(_BYTE *)(a1 + 160);
    v18 = v17;
    if ( v17 )
    {
      sub_2AAAFA0(&v18);
      v19 = v18;
      if ( v18 )
      {
        sub_2AAAFA0(&v19);
        v21 = v19;
        if ( v19 )
        {
          sub_2AAAFA0(&v21);
          v20 = v2;
          v22[0] = v21;
          if ( v21 )
            sub_2AAAFA0(v22);
          goto LABEL_9;
        }
LABEL_17:
        v20 = v2;
        v22[0] = 0;
LABEL_9:
        sub_2AAF310(v5, 34, &v20, 1, v22, v4);
        sub_9C6650(v22);
        sub_2BF0340(v5 + 96, 1, v15, v5, v6, v7);
        *(_QWORD *)v5 = &unk_4A231C8;
        *(_QWORD *)(v5 + 40) = &unk_4A23200;
        *(_QWORD *)(v5 + 96) = &unk_4A23238;
        sub_9C6650(&v21);
        *(_QWORD *)v5 = &unk_4A23FE8;
        *(_QWORD *)(v5 + 40) = &unk_4A24030;
        *(_QWORD *)(v5 + 96) = &unk_4A24068;
        sub_9C6650(&v19);
        v10 = *(unsigned int *)(v5 + 60);
        *(_QWORD *)v5 = &unk_4A232E8;
        *(_QWORD *)(v5 + 96) = &unk_4A23370;
        *(_QWORD *)(v5 + 40) = &unk_4A23338;
        *(_QWORD *)(v5 + 152) = v16;
        v11 = *(unsigned int *)(v5 + 56);
        if ( v11 + 1 > v10 )
        {
          sub_C8D5F0(v5 + 48, (const void *)(v5 + 64), v11 + 1, 8u, v8, v9);
          v11 = *(unsigned int *)(v5 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v5 + 48) + 8 * v11) = v3;
        ++*(_DWORD *)(v5 + 56);
        v12 = *(unsigned int *)(v3 + 24);
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v3 + 28) )
        {
          sub_C8D5F0(v3 + 16, (const void *)(v3 + 32), v12 + 1, 8u, v8, v9);
          v12 = *(unsigned int *)(v3 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8 * v12) = v5 + 40;
        ++*(_DWORD *)(v3 + 24);
        sub_9C6650(&v18);
        *(_QWORD *)v5 = &unk_4A24A10;
        *(_QWORD *)(v5 + 96) = &unk_4A24A98;
        *(_QWORD *)(v5 + 40) = &unk_4A24A60;
        *(_BYTE *)(v5 + 160) = v14;
        goto LABEL_14;
      }
    }
    else
    {
      v19 = 0;
    }
    v21 = 0;
    goto LABEL_17;
  }
LABEL_14:
  sub_9C6650(&v17);
  return v5;
}
