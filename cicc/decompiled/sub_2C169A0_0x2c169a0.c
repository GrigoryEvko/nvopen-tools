// Function: sub_2C169A0
// Address: 0x2c169a0
//
__int64 __fastcall sub_2C169A0(__int64 a1)
{
  __int64 v1; // r13
  __int64 *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r9
  __int64 v6; // r14
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v21; // [rsp+20h] [rbp-80h]
  __int64 v22; // [rsp+28h] [rbp-78h]
  __int64 v23; // [rsp+30h] [rbp-70h]
  __int64 v24; // [rsp+40h] [rbp-60h] BYREF
  __int64 v25; // [rsp+48h] [rbp-58h] BYREF
  __int64 v26; // [rsp+50h] [rbp-50h] BYREF
  __int64 v27; // [rsp+58h] [rbp-48h] BYREF
  __int64 v28; // [rsp+60h] [rbp-40h] BYREF
  __int64 v29[7]; // [rsp+68h] [rbp-38h] BYREF

  v1 = 0;
  v22 = *(_QWORD *)(a1 + 136);
  v2 = *(__int64 **)(a1 + 48);
  if ( *(_DWORD *)(a1 + 56) )
    v1 = *v2;
  v3 = v2[1];
  v4 = v2[2];
  v23 = *(_QWORD *)(a1 + 152);
  v24 = *(_QWORD *)(a1 + 88);
  if ( v24 )
    sub_2AAAFA0(&v24);
  v6 = sub_22077B0(0xA8u);
  if ( v6 )
  {
    v21 = *(_QWORD *)(a1 + 160);
    v25 = v24;
    if ( v24 )
    {
      sub_2AAAFA0(&v25);
      v26 = v25;
      if ( v25 )
      {
        sub_2AAAFA0(&v26);
        v28 = v26;
        if ( v26 )
        {
          sub_2AAAFA0(&v28);
          v27 = v1;
          v29[0] = v28;
          if ( v28 )
            sub_2AAAFA0(v29);
          goto LABEL_11;
        }
LABEL_23:
        v27 = v1;
        v29[0] = 0;
LABEL_11:
        sub_2AAF310(v6, 33, &v27, 1, v29, v5);
        sub_9C6650(v29);
        sub_2BF0340(v6 + 96, 1, v22, v6, v7, v8);
        v9 = v6 + 40;
        *(_QWORD *)v6 = &unk_4A231C8;
        *(_QWORD *)(v6 + 40) = &unk_4A23200;
        *(_QWORD *)(v6 + 96) = &unk_4A23238;
        sub_9C6650(&v28);
        *(_QWORD *)v6 = &unk_4A23FE8;
        *(_QWORD *)(v6 + 40) = &unk_4A24030;
        *(_QWORD *)(v6 + 96) = &unk_4A24068;
        sub_9C6650(&v26);
        v12 = *(unsigned int *)(v6 + 60);
        *(_QWORD *)v6 = &unk_4A232E8;
        *(_QWORD *)(v6 + 96) = &unk_4A23370;
        *(_QWORD *)(v6 + 40) = &unk_4A23338;
        *(_QWORD *)(v6 + 152) = v23;
        v13 = *(unsigned int *)(v6 + 56);
        if ( v13 + 1 > v12 )
        {
          sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v13 + 1, 8u, v10, v11);
          v13 = *(unsigned int *)(v6 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v13) = v3;
        ++*(_DWORD *)(v6 + 56);
        v14 = *(unsigned int *)(v3 + 24);
        if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v3 + 28) )
        {
          sub_C8D5F0(v3 + 16, (const void *)(v3 + 32), v14 + 1, 8u, v10, v11);
          v14 = *(unsigned int *)(v3 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8 * v14) = v9;
        ++*(_DWORD *)(v3 + 24);
        sub_9C6650(&v25);
        v17 = *(unsigned int *)(v6 + 60);
        *(_QWORD *)v6 = &unk_4A24088;
        *(_QWORD *)(v6 + 96) = &unk_4A24110;
        *(_QWORD *)(v6 + 40) = &unk_4A240D8;
        *(_QWORD *)(v6 + 160) = v21;
        v18 = *(unsigned int *)(v6 + 56);
        if ( v18 + 1 > v17 )
        {
          sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v18 + 1, 8u, v15, v16);
          v18 = *(unsigned int *)(v6 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v18) = v4;
        ++*(_DWORD *)(v6 + 56);
        v19 = *(unsigned int *)(v4 + 24);
        if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 28) )
        {
          sub_C8D5F0(v4 + 16, (const void *)(v4 + 32), v19 + 1, 8u, v15, v16);
          v19 = *(unsigned int *)(v4 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8 * v19) = v9;
        ++*(_DWORD *)(v4 + 24);
        goto LABEL_20;
      }
    }
    else
    {
      v26 = 0;
    }
    v28 = 0;
    goto LABEL_23;
  }
LABEL_20:
  sub_9C6650(&v24);
  return v6;
}
