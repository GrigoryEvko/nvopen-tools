// Function: sub_2ABB2E0
// Address: 0x2abb2e0
//
__int64 __fastcall sub_2ABB2E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // [rsp+0h] [rbp-A0h]
  __int64 v25; // [rsp+0h] [rbp-A0h]
  __int64 v26; // [rsp+28h] [rbp-78h]
  __int64 v27; // [rsp+30h] [rbp-70h]
  __int64 v28; // [rsp+38h] [rbp-68h]
  __int64 v29; // [rsp+48h] [rbp-58h] BYREF
  __int64 v30; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+58h] [rbp-48h] BYREF
  __int64 v32; // [rsp+60h] [rbp-40h] BYREF
  __int64 v33[7]; // [rsp+68h] [rbp-38h] BYREF

  v27 = sub_2C47690(a5, *(_QWORD *)(a4 + 32), a6);
  v26 = a5 + 272;
  if ( *(_BYTE *)a2 != 67 )
  {
    v29 = *(_QWORD *)(a1 + 48);
    if ( v29 )
      sub_2AAAFA0(&v29);
    v9 = sub_22077B0(0xA8u);
    if ( !v9 )
      goto LABEL_15;
    v30 = v29;
    if ( v29 )
    {
      sub_2AAAFA0(&v30);
      v31 = v30;
      if ( v30 )
      {
        sub_2AAAFA0(&v31);
        v32 = v31;
        if ( v31 )
          sub_2AAAFA0(&v32);
        goto LABEL_26;
      }
    }
    else
    {
      v31 = 0;
    }
    v32 = 0;
LABEL_26:
    v33[0] = a3;
    sub_2ABB100(v9, 33, v33, 1, a1, &v32);
    sub_9C6650(&v32);
    *(_QWORD *)v9 = &unk_4A23FE8;
    *(_QWORD *)(v9 + 40) = &unk_4A24030;
    *(_QWORD *)(v9 + 96) = &unk_4A24068;
    sub_9C6650(&v31);
    *(_QWORD *)(v9 + 152) = a4;
    *(_QWORD *)v9 = &unk_4A232E8;
    *(_QWORD *)(v9 + 40) = &unk_4A23338;
    *(_QWORD *)(v9 + 96) = &unk_4A23370;
    sub_2AAECA0(v9 + 40, v27, (__int64)&unk_4A23338, v18, v19, v20);
    sub_9C6650(&v30);
    *(_QWORD *)(v9 + 160) = 0;
    *(_QWORD *)v9 = &unk_4A24088;
    *(_QWORD *)(v9 + 40) = &unk_4A240D8;
    *(_QWORD *)(v9 + 96) = &unk_4A24110;
    sub_2AAECA0(v9 + 40, v26, (__int64)&unk_4A240D8, v21, v22, v23);
    goto LABEL_15;
  }
  v29 = *(_QWORD *)(a2 + 48);
  if ( v29 )
    sub_2AAAFA0(&v29);
  v9 = sub_22077B0(0xA8u);
  if ( v9 )
  {
    v30 = v29;
    if ( v29 )
    {
      sub_2AAAFA0(&v30);
      v31 = v30;
      if ( v30 )
      {
        sub_2AAAFA0(&v31);
        v32 = v31;
        if ( v31 )
        {
          sub_2AAAFA0(&v32);
          v33[0] = v32;
          if ( v32 )
            sub_2AAAFA0(v33);
          goto LABEL_10;
        }
LABEL_18:
        v33[0] = 0;
LABEL_10:
        v10 = *(unsigned int *)(a3 + 24);
        *(_QWORD *)(v9 + 24) = 0;
        v11 = v9 + 40;
        *(_QWORD *)(v9 + 32) = 0;
        *(_QWORD *)(v9 + 64) = a3;
        *(_QWORD *)v9 = &unk_4A231A8;
        *(_BYTE *)(v9 + 8) = 33;
        *(_QWORD *)(v9 + 16) = 0;
        *(_QWORD *)(v9 + 40) = &unk_4A23170;
        *(_QWORD *)(v9 + 48) = v9 + 64;
        *(_QWORD *)(v9 + 56) = 0x200000001LL;
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 28) )
        {
          sub_C8D5F0(a3 + 16, (const void *)(a3 + 32), v10 + 1, 8u, v11, v10 + 1);
          v10 = *(unsigned int *)(a3 + 24);
          v11 = v9 + 40;
        }
        *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v10) = v11;
        ++*(_DWORD *)(a3 + 24);
        *(_QWORD *)(v9 + 80) = 0;
        *(_QWORD *)(v9 + 40) = &unk_4A23AA8;
        v12 = v33[0];
        *(_QWORD *)v9 = &unk_4A23A70;
        *(_QWORD *)(v9 + 88) = v12;
        if ( v12 )
        {
          v24 = v11;
          sub_2AAAFA0((__int64 *)(v9 + 88));
          v11 = v24;
        }
        v25 = v11;
        sub_9C6650(v33);
        sub_2BF0340(v9 + 96, 1, a1, v9);
        *(_QWORD *)v9 = &unk_4A231C8;
        *(_QWORD *)(v9 + 40) = &unk_4A23200;
        *(_QWORD *)(v9 + 96) = &unk_4A23238;
        sub_9C6650(&v32);
        *(_QWORD *)v9 = &unk_4A23FE8;
        *(_QWORD *)(v9 + 40) = &unk_4A24030;
        *(_QWORD *)(v9 + 96) = &unk_4A24068;
        sub_9C6650(&v31);
        *(_QWORD *)(v9 + 152) = a4;
        v28 = v25;
        *(_QWORD *)v9 = &unk_4A232E8;
        *(_QWORD *)(v9 + 40) = &unk_4A23338;
        *(_QWORD *)(v9 + 96) = &unk_4A23370;
        sub_2AAECA0(v28, v27, (__int64)&unk_4A23338, v13, v28, v14);
        sub_9C6650(&v30);
        *(_QWORD *)(v9 + 160) = a2;
        *(_QWORD *)v9 = &unk_4A24088;
        *(_QWORD *)(v9 + 40) = &unk_4A240D8;
        *(_QWORD *)(v9 + 96) = &unk_4A24110;
        sub_2AAECA0(v28, v26, (__int64)&unk_4A240D8, v15, v28, v16);
        goto LABEL_15;
      }
    }
    else
    {
      v31 = 0;
    }
    v32 = 0;
    goto LABEL_18;
  }
LABEL_15:
  sub_9C6650(&v29);
  return v9;
}
