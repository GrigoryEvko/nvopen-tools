// Function: sub_2C274A0
// Address: 0x2c274a0
//
__int64 __fastcall sub_2C274A0(__int64 a1, int a2, __int64 *a3, __int64 a4, __int64 **a5, __int64 *a6)
{
  int v6; // r12d
  __int64 *v8; // r14
  __int64 v9; // r8
  __int64 *v10; // r11
  __int64 v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // eax
  char v21; // dl
  __int64 result; // rax
  __int64 v25; // [rsp+48h] [rbp-48h] BYREF
  __int64 v26; // [rsp+50h] [rbp-40h] BYREF
  __int64 v27[7]; // [rsp+58h] [rbp-38h] BYREF

  v6 = a2;
  v25 = *a6;
  if ( !v25 )
  {
    v26 = 0;
    v8 = a3;
    goto LABEL_20;
  }
  v8 = a3;
  sub_2C25AB0(&v25);
  v26 = v25;
  if ( !v25 )
  {
LABEL_20:
    v27[0] = 0;
    goto LABEL_5;
  }
  sub_2C25AB0(&v26);
  v27[0] = v26;
  if ( v26 )
    sub_2C25AB0(v27);
LABEL_5:
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  v9 = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  v10 = &a3[a4];
  *(_BYTE *)(a1 + 8) = 18;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  if ( v10 != a3 )
  {
    v11 = *a3;
    v12 = v8;
    v13 = a1 + 64;
    v14 = 0;
    v15 = v10;
    while ( 1 )
    {
      *(_QWORD *)(v13 + 8 * v14) = v11;
      ++*(_DWORD *)(a1 + 56);
      v16 = *(unsigned int *)(v11 + 24);
      if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 28) )
      {
        sub_C8D5F0(v11 + 16, (const void *)(v11 + 32), v16 + 1, 8u, v9, v16 + 1);
        v16 = *(unsigned int *)(v11 + 24);
      }
      ++v12;
      *(_QWORD *)(*(_QWORD *)(v11 + 16) + 8 * v16) = a1 + 40;
      ++*(_DWORD *)(v11 + 24);
      if ( v15 == v12 )
        break;
      v14 = *(unsigned int *)(a1 + 56);
      v11 = *v12;
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v14 + 1, 8u, v9, v14 + 1);
        v14 = *(unsigned int *)(a1 + 56);
      }
      v13 = *(_QWORD *)(a1 + 48);
    }
    v6 = a2;
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v17 = v27[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v17;
  if ( v17 )
    sub_2C25AB0((__int64 *)(a1 + 88));
  sub_9C6650(v27);
  sub_2BF0340(a1 + 96, 1, 0, a1, v18, v19);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v26);
  *(_BYTE *)(a1 + 152) = 7;
  *(_DWORD *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A23258;
  *(_QWORD *)(a1 + 40) = &unk_4A23290;
  *(_QWORD *)(a1 + 96) = &unk_4A232C8;
  sub_9C6650(&v25);
  *(_DWORD *)(a1 + 160) = v6;
  *(_QWORD *)a1 = &unk_4A23D28;
  *(_QWORD *)(a1 + 96) = &unk_4A23DA0;
  *(_QWORD *)(a1 + 40) = &unk_4A23D68;
  *(_QWORD *)(a1 + 168) = a5;
  v27[0] = sub_B612D0(*a5, v6);
  v20 = sub_A746F0(v27);
  v21 = (v20 >> 6) | (v20 >> 4) | v20 | (v20 >> 2);
  *(_BYTE *)(a1 + 176) = !(((v20 & 0x40) != 0) | ((v20 & 0x10) != 0) | v20 & 1 | ((v20 & 4) != 0));
  *(_BYTE *)(a1 + 177) = (v21 & 2) == 0;
  if ( (v21 & 2) != 0 && (unsigned __int8)sub_A73ED0(v27, 41) )
    result = (unsigned int)sub_A73ED0(v27, 76) ^ 1;
  else
    result = 1;
  *(_BYTE *)(a1 + 178) = result;
  return result;
}
